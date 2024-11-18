from typing import Dict
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from config import ModelConfig, TrainingConfig, DataConfig


def setup_model_and_tokenizer(model_config: ModelConfig, train_config: TrainingConfig) -> tuple:
    """Initialize the language model and tokenizer with specified configuration."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length, 
        dtype=model_config.dtype, 
        load_in_4bit=model_config.load_in_4bit, 
    )
    
    # Configure PEFT settings
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config.lora_r,
        target_modules=model_config.target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=train_config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Set chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    return model, tokenizer

def prepare_dataset(config: DataConfig, tokenizer) -> Dataset:
    """Load and prepare the dataset for training."""
    def format_conversations(examples: Dict) -> Dict:
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    # Load and preprocess data
    df = pd.read_json(config.data_path)
    df = df.dropna(subset=[config.data_code_column_name, config.data_code_explanation_column_name, config.data_rust_translation_column_name])
    
    # Create conversation format
    df['prompt'] = '[Fortran Code]' + df[config.data_code_column_name] + \
                  '[Fortran Code Explain]' + df[config.data_code_explanation_column_name]
    
    df['conversations'] = df.apply(
        lambda row: [
            {'content': row['prompt'], 'role': 'user'},
            {'content': row[config.data_rust_translation_column_name], 'role': 'assistant'}
        ],
        axis=1
    )
    
    df['source'] = "not_sure_yet" # TO_ASK
    df['score'] = "0.0" # TO_ASK

    df_reset = df[['conversations', 'source', 'score']].reset_index(drop=True)
    
    # Prepare final dataset
    dataset = Dataset.from_pandas(df_reset)
    
    return dataset.map(format_conversations, batched=True)

def setup_trainer(
    model,
    tokenizer,
    dataset,
    config: TrainingConfig
) -> SFTTrainer:
    """Configure and return the SFT trainer."""
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        report_to=config.report_to,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=config.dataset_num_proc,
        packing=config.packing,
        args=training_args,
    )
    
    return train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

def print_gpu_stats(trainer_stats=None, start_memory: float = None):
    """Print GPU memory usage and training statistics."""
    gpu_stats = torch.cuda.get_device_properties(0)
    current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"Current memory reserved = {current_memory} GB.")
    
    if trainer_stats and start_memory:
        memory_for_training = round(current_memory - start_memory, 3)
        total_percentage = round(current_memory / max_memory * 100, 3)
        training_percentage = round(memory_for_training / max_memory * 100, 3)
        
        print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
        print(f"Peak reserved memory: {current_memory} GB")
        print(f"Memory used for training: {memory_for_training} GB")
        print(f"Total memory usage: {total_percentage}%")
        print(f"Training memory usage: {training_percentage}%")

def main():
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config, training_config)
    
    # Record initial GPU memory
    initial_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print_gpu_stats()
    
    # Prepare dataset
    dataset = prepare_dataset(data_config, tokenizer)
    
    # Setup and run training
    trainer = setup_trainer(model, tokenizer, dataset, training_config)
    trainer_stats = trainer.train()
    
    # Print final statistics
    print_gpu_stats(trainer_stats, initial_memory)

    # Local saving
    model.save_pretrained("lora_model") 
    tokenizer.save_pretrained("lora_model")

    # # Online saving
    # model.push_to_hub("your_name/lora_model", token = "...") 
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") 

if __name__ == "__main__":
    main()
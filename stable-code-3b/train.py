from datasets import load_dataset

# Load data from text file
dataset = load_dataset('text', data_files='fortran_snippets.txt', split='train')

# Preprocess data
def preprocess(example):
    return {'input': '', 'target': example['text']}

dataset = dataset.map(preprocess, remove_columns=['text'])

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "stabilityai/stable-code-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3, 
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_steps=500,
)

def preprocess_function(examples):
    targets = examples['target']
    model_inputs = tokenizer(targets, max_length=1024, truncation=True)
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,  
    train_dataset=dataset,
)

trainer.train()

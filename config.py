from dataclasses import dataclass
from typing import List, Optional

# Hyperparameters for Model
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
lora_r = 16 # Number of attention heads for LoRA
lora_alpha = 16 # Alpha value for LoRA
lora_dropout = 0 # Dropout rate for LoRA

# Hyperparameters for Training
per_device_train_batch_size = 2 # Batch size per GPU
gradient_accumulation_steps = 4 # Accumulate gradients for multiple steps
warmup_steps = 5 # Warmup steps for learning rate scheduler
learning_rate = 2e-4 # Learning rate for training
weight_decay = 0.01 # Weight decay for training
lr_scheduler_type = "linear" # Type of learning rate scheduler, can be "linear" / "cosine"
random_seed = 3407 # Random seed for reproducibility

# Data Configuration
data_path = "hf://datasets/CodeTranslatorLLM/Code-Translation/final_responses.json" # Path to the dataset
data_code_column_name = 'program_code' # Column name for code ("Fortran" in our case. Changing this won't affect our training)
data_code_explanation_column_name = 'fortran_code_explanation' # Column name for code explanation
data_rust_translation_column_name = 'rust_code_translation' # Column name for Rust translation (target output)

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = max_seq_length 
    dtype: Optional[str] = dtype
    load_in_4bit: bool = load_in_4bit
    lora_r: int = lora_r
    lora_alpha: int = lora_alpha
    lora_dropout: float = lora_dropout
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    per_device_train_batch_size: int = per_device_train_batch_size
    gradient_accumulation_steps: int = gradient_accumulation_steps
    warmup_steps: int = warmup_steps
    max_steps: int = 60
    learning_rate: float = learning_rate
    weight_decay: float = weight_decay
    lr_scheduler_type: str = lr_scheduler_type
    logging_steps: int = 1
    output_dir: str = "outputs"
    seed: int = random_seed
    dataset_num_proc: int = 2
    packing: bool = False
    report_to: str = "none"
    max_seq_length: int = max_seq_length

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = data_path
    data_code_column_name: str = data_code_column_name
    data_code_explanation_column_name: str = data_code_explanation_column_name
    data_rust_translation_column_name: str = data_rust_translation_column_name
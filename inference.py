from unsloth import FastLanguageModel
from transformers import TextStreamer
from typing import Tuple, List, Dict
import torch

def load_model(
    model_name: str,
    max_seq_length: int,
    dtype: torch.dtype,
    load_in_4bit: bool
) -> Tuple[FastLanguageModel, any]:
    """
    Load and initialize the language model for inference.
    
    Args:
        model_name (str): Name of the pre-trained model to load
        max_seq_length (int): Maximum sequence length for the model
        dtype (torch.dtype): Data type for model weights
        load_in_4bit (bool): Whether to load model in 4-bit quantization
    
    Returns:
        Tuple[FastLanguageModel, any]: Tuple containing the model and tokenizer
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def prepare_input(
    messages: List[Dict[str, str]],
    tokenizer: any,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prepare input for the model by applying chat template and tokenization.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries
        tokenizer: The tokenizer instance
        device (str): Device to load tensors to ("cuda" or "cpu")
    
    Returns:
        torch.Tensor: Prepared input tensor
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

def generate_response(
    model: FastLanguageModel,
    inputs: torch.Tensor,
    tokenizer: any,
    max_new_tokens: int = 2000,
    temperature: float = 1.5,
    min_p: float = 0.1,
    skip_prompt: bool = True
) -> str:
    """
    Generate response using the model.
    
    Args:
        model (FastLanguageModel): The language model
        inputs (torch.Tensor): Prepared input tensor
        tokenizer: The tokenizer instance
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        min_p (float): Minimum probability for nucleus sampling
        skip_prompt (bool): Whether to skip prompt in output
    
    Returns:
        str: Generated response
    """
    text_streamer = TextStreamer(tokenizer, skip_prompt=skip_prompt)
    outputs = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p
    )
    return outputs

def main(
        USER_INPUT_CODE = "program sum_of_numbers\n    implicit none\n    integer :: n, i, sum\n\n    ! Initialize variables\n    sum = 0\n\n    ! Get user input\n    print *, \"Enter a positive integer:\"\n    read *, n\n\n    ! Calculate the sum of numbers from 1 to n\n    do i = 1, n\n        sum = sum + i\n    end do\n\n    ! Print the result\n    print *, \"The sum of numbers from 1 to\", n, \"is\", sum\nend program sum_of_numbers",
        USER_INPUT_EXPLANATION = "The provided Fortran code snippet is a program that calculates the sum of integers from 1 to n, where n is provided by the user. It uses a simple procedural approach, including variable declarations, input handling, and a loop for the summation.\n\nThe functionality of the program is explained in detail in the elaboration. The program starts by initializing variables and prompting the user for input. It then calculates the sum using a do loop, iterating from 1 to n, and accumulating the result in a variable. Finally, it prints the computed sum to the console.\n\nThis program demonstrates a straightforward application of Fortran's capabilities for handling loops and basic arithmetic operations. It is a clear example of how Fortran can be used to solve mathematical problems involving user interaction and iterative computations.",
        MODEL_PATH  = "lora_model"
        ):
    """
    Main function to demonstrate the inference pipeline.
    """
    # Import configuration
    from config import max_seq_length, dtype, load_in_4bit
    
    # Example messages
    messages = [
        {
            "role": "user",
            "content": str("[Fortran Code]") + str(USER_INPUT_CODE) + str("[Fortran Code Explain]") + str(USER_INPUT_EXPLANATION)
        }
    ]
    
    # Load model
    model, tokenizer = load_model(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    
    # Prepare input
    inputs = prepare_input(messages, tokenizer)
    
    # Generate response
    return generate_response(model, inputs, tokenizer)

if __name__ == "__main__":
    # YOUR_FORTRAN_CODE_HERE
    USER_INPUT_CODE = "program sum_of_numbers\n    implicit none\n    integer :: n, i, sum\n\n    ! Initialize variables\n    sum = 0\n\n    ! Get user input\n    print *, \"Enter a positive integer:\"\n    read *, n\n\n    ! Calculate the sum of numbers from 1 to n\n    do i = 1, n\n        sum = sum + i\n    end do\n\n    ! Print the result\n    print *, \"The sum of numbers from 1 to\", n, \"is\", sum\nend program sum_of_numbers" 
    
    # YOUR_EXPLANATION_HERE
    USER_INPUT_EXPLANATION = "The provided Fortran code snippet is a program that calculates the sum of integers from 1 to n, where n is provided by the user. It uses a simple procedural approach, including variable declarations, input handling, and a loop for the summation.\n\nThe functionality of the program is explained in detail in the elaboration. The program starts by initializing variables and prompting the user for input. It then calculates the sum using a do loop, iterating from 1 to n, and accumulating the result in a variable. Finally, it prints the computed sum to the console.\n\nThis program demonstrates a straightforward application of Fortran's capabilities for handling loops and basic arithmetic operations. It is a clear example of how Fortran can be used to solve mathematical problems involving user interaction and iterative computations."
    
    # YOUR_MODEL_PATH_HERE
    MODEL_PATH = "lora_model"
    
    main(USER_INPUT_CODE, USER_INPUT_EXPLANATION, MODEL_PATH)
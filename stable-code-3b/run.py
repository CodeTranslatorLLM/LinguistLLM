import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Path to your local directory containing the model and tokenizer files
local_directory = "/projectnb/ivc-ml/ssjoshi/stable-code-3b"  # Replace with the path to the directory from your screenshot

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_directory)
model = AutoModelForCausalLM.from_pretrained(local_directory, torch_dtype=torch.float16)

model.cuda()

# Generate tokens
inputs = tokenizer("import numpy", return_tensors="pt").to(model.device)
tokens = model.generate(
  **inputs,
  max_new_tokens=48,
  temperature=0.2,
  do_sample=True,
)

# Decode and print the generated code
generated_code = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(generated_code)

# # Specify the directory where you want to save the model and tokenizer
# my_model_directory = '/projectnb/ivc-ml/ssjoshi/stable-code-3b/'

# # Ensure the directory exists or create it
# import os
# if not os.path.exists(my_model_directory):
#     os.makedirs(my_model_directory)

# # Save the model and tokenizer
# model.save_pretrained(my_model_directory)
# tokenizer.save_pretrained(my_model_directory)





# tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-3b")
# model = AutoModelForCausalLM.from_pretrained(
#   "stabilityai/stable-code-3b",
#   torch_dtype="auto",
# )
# model.cuda()
# inputs = tokenizer("import torch\nimport torch.nn as nn", return_tensors="pt").to(model.device)
# tokens = model.generate(
#   **inputs,
#   max_new_tokens=48,
#   temperature=0.2,
#   do_sample=True,
# )
# print(tokenizer.decode(tokens[0], skip_special_tokens=True))

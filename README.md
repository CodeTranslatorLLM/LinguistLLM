# CodeConvLLM

Develop a Language Model (LLM) that can accurately translate OCaml and Fortran code into C# and Rust, integrate the model into a Visual Studio Code plugin, and ensure the translated code meets industry standards through benchmarking.

WIP Doc = **https://docs.google.com/document/d/1CVzw5MXcq6ky3k56w0bbe1-VUq6FrUC6rpfZmLhLMTM/edit?usp=sharing**

Add .env file with the perplexity key as PPX_API_KEY

# User Manual
reference: **https://github.com/unslothai/unsloth**

# Installation Instructions
install unsloth `"xformers==0.0.28.post2"` and "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" with:
```
make install
```
refer to **https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions** if error

# Train Model
run main.py with:
```
make run_train
```

##### Hyperparameter
Modify hyperparameters in **Hyperparameters** blocks in config.py

##### Dataset:=
Modify dataset in **Data Configuration** blocks in config.py

# Inference
run inference with:
```
make run_inference
```
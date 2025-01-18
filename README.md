# CodeConvLLM

Our team develop a Language Model (LLM) that can accurately translate Fortran code into Rust, integrate the model into an API, and ensure the translated code meets industry standards through benchmarking.

You can use this to train other code-to-code tranlation.

This repository leverages the knowledge distillation technique for training. In our process:

1. We utilize the [CodeTranslatorLLM/Code-Translation](https://huggingface.co/datasets/CodeTranslatorLLM/Code-Translation) dataset to generate translated Rust code.
2. The teacher model, GPT-4, is used to generate these translations.
3. We then fine-tune a smaller student model, Llama-3.2-3B-Instruct, using the train.py script provided in this repository.

This approach allows us to transfer knowledge from a larger, more powerful model to a smaller, more efficient model with only 3B parameters for code translation tasks.

# Dashboard Playground
We use Gradio to create an interactive graphical user interface (GUI) where users can:

+ Enter their Fortran code into a text block.
+ Generate the translated Rust code by clicking the translation button.

Run cells in `Gradio.ipynb` to interact with an AI model that translates your Fortran code to Rust code with our LoRA model!

# User Manual (Training and Inferencing)
small language mode: Llama-3.2-3B-Instruct
reference: **https://github.com/unslothai/unsloth**

# Installation Instructions
install all the requirement with
```
make install
```
If wanting to use **pip3** instead of pip, change `PIP := pip` to `PIP := pip3`

If wanting to instal manually:
```
pip install pandas
pip install unsloth `"xformers==0.0.28.post2"` and "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 
```
If failed to install unsloth, please refer to **https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions** 

# Train Model
Our training uses
+ Data: [CodeTranslatorLLM/Code-Translation](https://huggingface.co/datasets/CodeTranslatorLLM/Code-Translation)
+ Base Model:
+ Fine-tune Model: Llama-3.2-3B-Instruct
run train.py with:
```
make run_train
```
You can modify hyperparameters in **Hyperparameters** blocks in config.py

##### Dataset:
The default dataset used in this project is [CodeTranslatorLLM/Code-Translation](https://huggingface.co/datasets/CodeTranslatorLLM/Code-Translation).

If you want to use your own dataset, you can easily customize the configuration by modifying the following parameters in the Data Configuration section of `config.py`:

+ `data_path`: Path to your dataset file.
+ `data_code_explanation_column_name`: Name of the column containing the code explanations.
+ `data_rust_translation_column_name`: Name of the column containing the Rust translations.

# Inference
To perform inference, ensure the lora_model folder is located in your current working directory. This folder contains the necessary model files required for running the inference process.

Run the inference using the following command:
```
make run_inference
```

Customizing Inference Inputs
You can customize the inputs for inference by modifying the following variables in the inference.py script:

+ `USER_INPUT_CODE`: Specify the input code for translation.
+ `YOUR_EXPLANATION_HERE`: Provide any additional explanation or context (if applicable).
+ `MODEL_PATH`: Set the path to your desired model for inference.
These configurations allow you to tailor the inference process to your specific needs.

#### Using the Inference API
The `inferenc_API.py` script allows you to serve the model as a web API, making it easier to integrate with external applications.

##### Start the API Server
1. Install required dependencies:
```bash
pip install fastapi uvicorn
```
2. Run the API:
```bash
python inferenc_API.py
```
3. The server will start on http://127.0.0.1:8000

##### Endpoints

**POST /generate_response/**
This endpoint accepts Fortran code and its explanation as input and returns the generated response from the model.
+**Request Body:**
```json
{
  "user_input_code": "Your Fortran code here",
  "user_input_explanation": "Explanation of the code here"
}
```

+**Response:**
```json
{
  "response": "Generated response from the model"
}
```

#### Example Usage with curl
Send a POST request to the API using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/generate_response/" \
-H "Content-Type: application/json" \
-d '{
  "user_input_code": "program sum_of_numbers\n    implicit none\n    integer :: n, i, sum\n    ...",
  "user_input_explanation": "This code calculates the sum of integers from 1 to n."
}'
```

**Example Response**
```json
{
  "response": "This program sums all integers from 1 to n, where n is provided by the user. It uses a loop to calculate the total."
}
```

**Integration with Applications**
You can integrate this API with tools like Postman, cURL, or custom front-end/back-end systems to interact with the model programmatically.


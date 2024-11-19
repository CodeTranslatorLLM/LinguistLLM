# CodeConvLLM

Develop a Language Model (LLM) that can accurately translate OCaml and Fortran code into C# and Rust, integrate the model into a Visual Studio Code plugin, and ensure the translated code meets industry standards through benchmarking.

WIP Doc = **https://docs.google.com/document/d/1CVzw5MXcq6ky3k56w0bbe1-VUq6FrUC6rpfZmLhLMTM/edit?usp=sharing**

Add .env file with the perplexity key as PPX_API_KEY

# Dashboard Playground
Run cells in Gradio to interact with an AI model that translates your Fortran code to Rust code with our LoRA model!

# User Manual (Training and Inferencing)
small language mode: Llama-3.2-3B-Instruct
reference: **https://github.com/unslothai/unsloth**

# Installation Instructions
install unsloth `"xformers==0.0.28.post2"` and "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" with:
```
make install
```
refer to **https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions** if error

# Train Model
run train.py with:
```
make run_train
```

##### Hyperparameter
Modify hyperparameters in **Hyperparameters** blocks in config.py

##### Dataset:
Modify dataset in **Data Configuration** blocks in config.py

# Inference
**Make sure "lora_model" folder is in your current folder for inferencing**

run inference with:
```
make run_inference
```

change the input of `USER_INPUT_CODE`, `YOUR_EXPLANATION_HERE`, `MODEL_PATH` in `inference.py` to customize your input 

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


from fastapi import FastAPI
from pydantic import BaseModel
import torch
import inference

# Assuming your existing code
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Define input schema
class InferenceRequest(BaseModel):
    user_input_code: str
    user_input_explanation: str

# Initialize FastAPI
app = FastAPI()

# Define the inference route
@app.post("/generate_response/")
async def generate_response_endpoint(request: InferenceRequest):
    # Extract input data from request
    user_input_code = request.user_input_code
    user_input_explanation = request.user_input_explanation
    
    # Prepare input for model (use your existing `prepare_input` and `generate_response` logic)
    model, tokenizer = inference.load_model(model_name="your_model", max_seq_length=128, dtype=torch.float32, load_in_4bit=False)
    
    # Example messages for your pipeline
    messages = [{"role": "user", "content": f"[Fortran Code]{user_input_code}[Fortran Code Explain]{user_input_explanation}"}]
    inputs = inference.prepare_input(messages, tokenizer)
    
    # Generate response using the model
    response = inference.generate_response(model, inputs, tokenizer)
    
    # Return the response as JSON
    return {"response": response}

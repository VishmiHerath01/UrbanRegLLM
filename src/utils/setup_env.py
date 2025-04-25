"""
Environment setup for Legal LLM fine-tuning project
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_environment():
    """Setup the environment for LLM fine-tuning"""
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    print(f"Using device: {device}")
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create necessary directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    return device

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.3", device="cpu", quantization=None):
    """
    Load the Mistral model and tokenizer
    
    Args:
        model_name (str): Name or path of the model
        device (str): Device to load the model on
        quantization (str): Quantization type (None, '4bit', '8bit')
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {model_name}...")
    
    # Configure quantization if specified
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if device != "cpu":
            model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure the tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully")
    return model, tokenizer

if __name__ == "__main__":
    device = setup_environment()
    # Test loading with a smaller model for verification
    model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.3", device, quantization="4bit")
    print("Setup completed successfully")

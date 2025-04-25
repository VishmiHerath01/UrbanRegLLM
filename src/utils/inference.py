"""
Inference module for Legal LLM
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_fine_tuned_model(model_path, base_model_name=None, device="cpu"):
    """
    Load a fine-tuned model
    
    Args:
        model_path (str): Path to the fine-tuned model
        base_model_name (str): Name of the base model if needed
        device (str): Device to load the model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Try to load as a merged model first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
        print(f"Loaded merged model from {model_path}")
        
    except Exception as e:
        print(f"Could not load as merged model, trying as adapter: {e}")
        if not base_model_name:
            base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            print(f"No base model specified, defaulting to {base_model_name}")
            
        # Load the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
        
        # Load the adapter
        model = PeftModel.from_pretrained(model, model_path)
        print(f"Loaded base model {base_model_name} with adapter from {model_path}")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """
    Generate a response from the model
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling
        
    Returns:
        str: Generated response
    """
    # Format the input as an instruction
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    
    # Decode and return only the new tokens (the response)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the response part (after [/INST])
    response_start = full_response.find("[/INST]") + len("[/INST]")
    response = full_response[response_start:].strip()
    
    # Remove ending token if present
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def interactive_qa(model, tokenizer, category=None):
    """
    Interactive Q&A session with the model
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        category (str): Optional legal category to focus on
    """
    print("=" * 50)
    print("Legal Assistant Interactive Q&A")
    print("=" * 50)
    if category:
        print(f"Category: {category}")
    print("Type 'exit' to end the session.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYour question: ")
        if user_input.lower() == "exit":
            break
        
        # Format with category if provided
        if category:
            prompt = f"You are a legal assistant specialized in urban planning and building regulations. Based on the following category: {category}\n\nQuestion: {user_input}"
        else:
            prompt = f"You are a legal assistant specialized in urban planning and building regulations. Answer the following question: {user_input}"
        
        # Generate and print response
        print("\nGenerating response...\n")
        response = generate_response(model, tokenizer, prompt)
        print(f"Legal Assistant: {response}")
        print("-" * 50)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Legal LLM Inference")
    
    parser.add_argument("--model_path", type=str, default="models/legal-mistral",
                        help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Base model name for adapter loading")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--category", type=str, default=None,
                        help="Legal category to focus on")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate response for")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_fine_tuned_model(args.model_path, args.base_model, args.device)
    
    # Run in interactive mode or single prompt mode
    if args.interactive:
        interactive_qa(model, tokenizer, args.category)
    elif args.prompt:
        response = generate_response(model, tokenizer, args.prompt)
        print(f"\nResponse: {response}")
    else:
        print("Please specify either --interactive or --prompt")

if __name__ == "__main__":
    main()

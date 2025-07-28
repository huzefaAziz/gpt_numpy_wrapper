import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    GPT2Config,
    PreTrainedModel,
    GenerationMixin
)
import torch

class M:
    """Base class with where method using numpy.where"""
    
    def where(self, condition, x=None, y=None):
        """
        Apply numpy.where with given condition and values
        
        Args:
            condition: array_like, bool condition
            x: array_like, optional - values to use where condition is True
            y: array_like, optional - values to use where condition is False
            
        Returns:
            ndarray or tuple of ndarrays
        """
        return np.where(condition, x, y)
    
    def __str__(self):
        return f"{self.__class__.__name__} instance"
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class gpt(M):
    """Custom GPT model wrapper that inherits numpy operations from M"""
    
    def __init__(self, model_name="gpt2", device="auto"):
        """
        Initialize GPT model with tokenizer
        
        Args:
            model_name: str - HuggingFace model name (default: "gpt2")
            device: str - device to load model on ("auto", "cpu", "cuda")
        """
        super().__init__()
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        
    def generate_text(self, prompt, max_length=100, temperature=0.7, num_return_sequences=1):
        """
        Generate text from a prompt
        
        Args:
            prompt: str - input text prompt
            max_length: int - maximum length of generated text
            temperature: float - sampling temperature
            num_return_sequences: int - number of sequences to generate
            
        Returns:
            list of generated text strings
        """
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def get_embeddings(self, text):
        """
        Get token embeddings for input text
        
        Args:
            text: str or list of str - input text(s)
            
        Returns:
            torch.Tensor - embeddings tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state (embeddings)
            embeddings = outputs.hidden_states[-1]
        
        return embeddings
    
    def mask_tokens(self, tokens, mask_prob=0.15):
        """
        Apply masking to tokens using numpy.where from parent class
        
        Args:
            tokens: array_like - input tokens
            mask_prob: float - probability of masking each token
            
        Returns:
            tuple: (masked_tokens, mask_positions)
        """
        # Convert to numpy if needed
        if hasattr(tokens, 'numpy'):
            tokens = tokens.numpy()
        tokens = np.array(tokens)
        
        # Create random mask
        mask_condition = np.random.rand(*tokens.shape) < mask_prob
        
        # Use inherited where method to apply masking
        masked_tokens = self.where(
            mask_condition, 
            self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else self.tokenizer.unk_token_id,
            tokens
        )
        
        return masked_tokens, mask_condition
    
    def filter_tokens(self, tokens, condition):
        """
        Filter tokens based on condition using numpy.where
        
        Args:
            tokens: array_like - input tokens
            condition: array_like, bool - filtering condition
            
        Returns:
            ndarray - filtered tokens
        """
        return self.where(condition, tokens, 0)
    
    def __str__(self):
        return f"gpt(model='{self.model_name}', device='{self.device}')"
    
    def __repr__(self):
        return f"gpt(model_name='{self.model_name}', device='{self.device}')"

# Example usage
if __name__ == "__main__":
    # Create GPT instance
    model = gpt("gpt2")
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    generated = model.generate_text(prompt, max_length=50)
    print("Generated text:")
    for text in generated:
        print(f"- {text}")
    
    # Example of using numpy operations
    tokens = np.array([1, 2, 3, 4, 5])
    condition = tokens > 3
    filtered = model.filter_tokens(tokens, condition)
    print(f"\nFiltered tokens: {filtered}")
    
    # Example of token masking
    masked_tokens, mask_pos = model.mask_tokens(tokens, mask_prob=0.3)
    print(f"Original tokens: {tokens}")
    print(f"Masked tokens: {masked_tokens}")
    print(f"Mask positions: {mask_pos}")
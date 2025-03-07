from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, Optional
import streamlit as st

torch.classes.__path__ = [] # add this line to manually set it to empty. 

@st.cache_resource
def load_cached_model(name: str, device_map: str = "cpu"):
    """
    Cached model loading function that persists the model across reruns
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_safetensors=True,
        use_flash_attention_2=False,
        use_cache=True,
        load_in_8bit=True,
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype=torch.float32,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    
    return tokenizer, model, pipe

class HuggingFaceGemmaModel:
    """
    A class for the Hugging Face Gemma model. Handles model selection, loading, and inference.
    Uses transformers pipeline for better text generation and formatting.

    Example
    -------
    Select Gemma 2B, 7B etc.
    
    Additional Information:
    ----------------------
    Complete Information: https://huggingface.co/google/gemma-2b

    Available Models:
    - google/gemma-2b (2B parameters, base)
    - google/gemma-2b-it (2B parameters, instruction-tuned)
    - google/gemma-7b (7B parameters, base)
    - google/gemma-7b-it (7B parameters, instruction-tuned)
    """
    
    AVAILABLE_MODELS: Dict[str, Dict] = {
        "gemma-2b": {
            "name": "google/gemma-2b",
            "description": "2B parameters, base model",
            "type": "base"
        },
        "gemma-2b-it": {
            "name": "google/gemma-2b-it",
            "description": "2B parameters, instruction-tuned",
            "type": "instruct"
        },
        "gemma-7b": {
            "name": "google/gemma-7b",
            "description": "7B parameters, base model",
            "type": "base"
        },
        "gemma-7b-it": {
            "name": "google/gemma-7b-it",
            "description": "7B parameters, instruction-tuned",
            "type": "instruct"
        }
    }
    
    def __init__(self, name: str = "google/gemma-2b"):
        self.name = name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self, device_map: str = "cpu"):
        """
        Load the model using cached resources
        
        Args:
            device_map: Device mapping strategy (should be "cpu" for CPU-only inference)
        """
        # Load cached model, tokenizer and pipeline
        self.tokenizer, self.model, self.pipeline = load_cached_model(self.name, device_map)
        return self
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **kwargs
    ) -> str:
        """
        Generate a response using the text generation pipeline
        
        Args:
            prompt: Input text
            max_length: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more creative)
            num_return_sequences: Number of responses to generate
            **kwargs: Additional generation parameters for the pipeline
        
        Returns:
            str: Generated response
        """
        if not self.pipeline:
            self.load_model()
        
        # Update generation config with any provided kwargs
        generation_config = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
            "do_sample": True,
            **kwargs
        }
        
        # Generate response using the pipeline
        outputs = self.pipeline(
            prompt,
            **generation_config
        )
        
        # Extract the generated text
        if num_return_sequences == 1:
            response = outputs[0]["generated_text"]
        else:
            # Join multiple sequences if requested
            response = "\n---\n".join(output["generated_text"] for output in outputs)
        
        return response.strip()
    
    def get_model_info(self) -> Dict:
        """Return information about the model"""
        return {
            "name": self.name,
            "loaded": self.model is not None,
            "pipeline_ready": self.pipeline is not None
        }
    
    def get_model_name(self) -> str:
        """Return the name of the model"""
        return self.name
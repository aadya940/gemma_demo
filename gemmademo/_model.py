import os
from typing import Dict
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class LlamaCppGemmaModel:
    """
    A class for the Gemma model using llama.cpp. This class replicates the API of the original
    HuggingFaceGemmaModel but uses llama.cpp for inference. It handles model selection, loading,
    downloading (if necessary), and text generation.

    Available Models (ensure the repo_id and filename match the GGUF file on Hugging Face):
    - gemma-2b: 2B parameters, base model
    - gemma-2b-it: 2B parameters, instruction-tuned
    - gemma-7b: 7B parameters, base model
    - gemma-7b-it: 7B parameters, instruction-tuned

    All models will be stored in the "models/" directory.
    """

    # Class variable to cache loaded models
    _model_cache = {}

    AVAILABLE_MODELS: Dict[str, Dict] = {
        "gemma-3b": {
            "model_path": "models/gemma-3-1b-it-Q5_K_M.gguf",
            "repo_id": "bartowski/google_gemma-3-1b-it-GGUF",  # Updated repo
            "filename": "google_gemma-3-1b-it-Q5_K_M.gguf",  # Better quantization
            "description": "3B parameters, instruction-tuned (Q5_K_M)",
            "type": "instruct",
        },
        "gemma-3b-q6": {
            "model_path": "models/gemma-3-1b-it-Q6_K.gguf",
            "repo_id": "bartowski/google_gemma-3-1b-it-GGUF",  # Updated repo
            "filename": "google_gemma-3-1b-it-Q6_K.gguf",  # Higher quality quantization
            "description": "3B parameters, instruction-tuned (Q6_K)",
            "type": "instruct",
        },
        "gemma-2b": {
            "model_path": "models/gemma-2b.gguf",
            "repo_id": "rahuldshetty/gemma-2b-gguf-quantized",  # update to the actual repo id
            "filename": "gemma-2b-Q4_K_M.gguf",  # update to the actual filename
            "description": "2B parameters, base model",
            "type": "base",
        },
        "gemma-2b-it": {
            "model_path": "models/gemma-2b-it.gguf",
            "repo_id": "MaziyarPanahi/gemma-2b-it-GGUF",  # update to the actual repo id
            "filename": "gemma-2b-it.Q4_K_M.gguf",  # update to the actual filename
            "description": "2B parameters, instruction-tuned",
            "type": "instruct",
        },
        "gemma-7b-it": {
            "model_path": "models/gemma-7b-it.gguf",
            "repo_id": "MaziyarPanahi/gemma-7b-GGUF",  # update to the actual repo id
            "filename": "gemma-7b.Q4_K_M.gguf",  # update to the actual filename
            "description": "7B parameters, instruction-tuned",
            "type": "instruct",
        },
        "gemma-7b-gguf": {
            "model_path": "models/gemma-7b.gguf",
            "repo_id": "rahuldshetty/gemma-7b-it-gguf-quantized",  # repository for the GGUF model
            "filename": "gemma-7b-it-Q4_K_M.gguf",  # updated filename for GGUF model
            "description": "7B parameters in GGUF format",
            "type": "base",
        },
    }

    def __init__(self, name: str = "gemma-3b"):
        """
        Initialize the model instance.

        Args:
            name (str): The model name (should match one of the AVAILABLE_MODELS keys).
        """
        self.name = name
        self.model = None  # Instance of Llama from llama.cpp
        self.messages = []

    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Load the model. If the model file does not exist, it will be downloaded.
        Uses caching to avoid reloading models unnecessarily.

        Args:
            n_ctx (int): Context window size.
            n_gpu_layers (int): Number of layers to offload to GPU (if supported; 0 for CPU-only).
        """
        # Check if model is already loaded in cache
        cache_key = f"{self.name}_{n_ctx}_{n_gpu_layers}"
        if cache_key in LlamaCppGemmaModel._model_cache:
            self.model = LlamaCppGemmaModel._model_cache[cache_key]
            return self

        model_info = self.AVAILABLE_MODELS.get(self.name)
        if not model_info:
            raise ValueError(f"Model {self.name} is not available.")

        model_path = model_info["model_path"]

        # If the model file doesn't exist, download it.
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            repo_id = model_info.get("repo_id")
            filename = model_info.get("filename")

            if repo_id is None or filename is None:
                raise ValueError(
                    "Repository ID or filename is missing for model download."
                )

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(model_path),
                local_dir_use_symlinks=False,
            )

            if downloaded_path != model_path:
                os.rename(downloaded_path, model_path)

        _threads = min(2, os.cpu_count() or 1)

        self.model = Llama(
            model_path=model_path,
            n_threads=_threads,
            n_threads_batch=_threads,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=8,
            verbose=False,  # Disable verbose output for better performance
        )

        # Cache the model for future use
        LlamaCppGemmaModel._model_cache[cache_key] = self.model
        return self

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ):
        """
        Generate a response using the llama.cpp model with optimized parameters.

        Args:
            prompt (str): Input prompt text.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (higher = more creative).
            top_p (float): Nucleus sampling threshold.
            top_k (int): Limit vocabulary choices to top K tokens.
            repeat_penalty (float): Penalize repeated words.

        Yields:
            str: Generated response text as a stream.
        """
        if self.model is None:
            self.load_model()

        self.messages.append({"role": "user", "content": prompt})

        response_stream = self.model.create_chat_completion(
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        self.messages.append({"role": "assistant", "content": ""})

        outputs = ""
        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                outputs += delta["content"]
                self.messages[-1]["content"] += delta["content"]
                yield outputs

    def get_model_info(self) -> Dict:
        """
        Return information about the model.

        Returns:
            Dict: A dictionary containing the model name and load status.
        """
        return {"name": self.name, "loaded": self.model is not None}

    def get_model_name(self) -> str:
        """
        Return the name of the model.

        Returns:
            str: Model name.
        """
        return self.name

from ._chat import GradioChat
from ._model import HuggingFaceGemmaModel
from ._prompts import PromptManager
import torch

def main():
    # Set torch settings for better performance
    torch.set_grad_enabled(False)  # Disable gradient computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache
    
    # Initialize the model with optimized settings
    model = HuggingFaceGemmaModel("google/gemma-2b").load_model()
    
    # Initialize the prompt manager
    prompt_manager = PromptManager(task="question_answering")
    
    # Initialize and launch the chat interface
    chat = GradioChat(model, prompt_manager)
    chat.launch()

if __name__ == "__main__":
    main() 
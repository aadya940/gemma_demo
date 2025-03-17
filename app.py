import gradio as gr
from gemmademo import LlamaCppGemmaModel, GradioChat, PromptManager

def main():
    # Model and task selection
    model_options = list(LlamaCppGemmaModel.AVAILABLE_MODELS.keys())
    task_options = ["Question Answering", "Text Generation", "Code Completion"]
    
    chat = GradioChat(model_options=model_options, task_options=task_options)
    chat.run()

if __name__ == "__main__":
    main()

import gradio as gr
from gemmademo import LlamaCppGemmaModel, GradioChat, PromptManager

def main():
    # Model and task selection
    model_options = list(LlamaCppGemmaModel.AVAILABLE_MODELS.keys())
    task_options = ["Question Answering", "Text Generation", "Code Completion"]
    
    def update_chat(model_name, task_name):
        model = LlamaCppGemmaModel(name=model_name)
        model.load_model()
        prompt_manager = PromptManager(task=task_name)
        chat = GradioChat(model=model, prompt_manager=prompt_manager)
        chat.run()
    
    gr.Interface(
        fn=update_chat,
        inputs=[
            gr.Dropdown(choices=model_options, value="gemma-2b-it", label="Select Gemma Model"),
            gr.Dropdown(choices=task_options, value="Question Answering", label="Select Task"),
        ],
        outputs=[],
    ).launch()

if __name__ == "__main__":
    main()

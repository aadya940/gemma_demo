import gradio as gr
from ._model import LlamaCppGemmaModel
from ._prompts import PromptManager


class GradioChat:
    """
    A class that handles the chat interface for the Gemma model.

    Features:
    - A Gradio-based chatbot UI.
    - Dynamically loads models based on user selection.
    - Uses Gemma (llama.cpp) for generating responses.
    - Formats user inputs before sending them to the model.
    """

    def __init__(self, prompt_manager: PromptManager, model_options: list[str], task_options: list[str]):
        self.prompt_manager = prompt_manager
        self.model_options = model_options
        self.task_options = task_options
        self.current_model_name = "gemma-2b-it"  # Default model
        self.model = self._load_model(self.current_model_name)

    def _load_model(self, model_name: str):
        """Loads the model dynamically when switching models."""
        return LlamaCppGemmaModel(name=model_name).load_model()

    def _chat(self):
        def chat_fn(message, history, selected_model, selected_task):
            if selected_model != self.current_model_name:
                self.current_model_name = selected_model
                self.model = self._load_model(selected_model)  # Reload model when changed

            prompt = self.prompt_manager.get_prompt(user_input=message, task=selected_task)
            response = self.model.generate_response(prompt)
            return response

        chat_interface = gr.ChatInterface(
            chat_fn,
            textbox=gr.Textbox(placeholder="Ask me something...", container=False),
            additional_inputs=[
                gr.Dropdown(choices=self.model_options, value=self.current_model_name, label="Select Gemma Model"),
                gr.Dropdown(choices=self.task_options, value="Question Answering", label="Select Task"),
            ],
        )
        chat_interface.launch()

    def run(self):
        self._chat()

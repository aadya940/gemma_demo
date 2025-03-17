import gradio as gr
from ._model import LlamaCppGemmaModel
from ._prompts import PromptManager


class GradioChat:
    """
    A class that handles the chat interface for the Gemma model.

    Features:
    - A Gradio-based chatbot UI.
    - Maintains chat history automatically.
    - Uses Gemma (Hugging Face) model for generating responses.
    - Formats user inputs before sending them to the model.
    """

    def __init__(self, model: LlamaCppGemmaModel, prompt_manager: PromptManager):
        self.model = model
        self.prompt_manager = prompt_manager

    def run(self):
        self._chat()

    def _chat(self):
        def chat_fn(history, message):
            prompt = self.prompt_manager.get_prompt(user_input=message)
            response = self.model.generate_response(prompt)
            return response

        chat_interface = gr.ChatInterface(
            chat_fn,
            textbox=gr.Textbox(placeholder="What is up?", container=False),
        )
        chat_interface.launch()

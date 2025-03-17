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

    def __init__(self, model: LlamaCppGemmaModel, prompt_manager: PromptManager, model_options: list[str], task_options: list[str]):
        self.model = model
        self.prompt_manager = prompt_manager
        self.model_options = model_options
        self.task_options = task_options

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
            additional_inputs=[
                gr.Dropdown(choices=self.model_options, value="gemma-2b-it", label="Select Gemma Model"),
                gr.Dropdown(choices=self.task_options, value="Question Answering", label="Select Task"),
            ],
        )
        chat_interface.launch()

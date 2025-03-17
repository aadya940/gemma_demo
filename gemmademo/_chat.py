import gradio as gr
from ._model import LlamaCppGemmaModel
from ._prompts import PromptManager


class GradioChat:
    """
    A class that handles the chat interface for the Gemma model.

    Features:
    - A Gradio-based chatbot UI.
    - Dynamically loads models based on user selection.
    - Dynamically updates tasks using PromptManager.
    - Uses Gemma (llama.cpp) for generating responses.
    """

    def __init__(self, model_options: list[str], task_options: list[str]):
        self.model_options = model_options
        self.task_options = task_options

        self.current_model_name = "gemma-3b"
        self.current_task_name = "Question Answering"

        # Load model lazily on first use instead of at initialization
        self.model = None
        self.prompt_manager = self._load_task(self.current_task_name)

        # Cache.
        self.models_cache = {}

    def _load_model(self, model_name: str):
        """Loads the model dynamically when switching models, with caching."""
        if model_name in self.models_cache:
            return self.models_cache[model_name]

        model = LlamaCppGemmaModel(name=model_name).load_model()
        self.models_cache[model_name] = model
        return model

    def _load_task(self, task_name: str):
        """Loads the task dynamically when switching tasks."""
        return PromptManager(task=task_name)

    def _chat(self):
        def chat_fn(message, history, selected_model, selected_task):
            # Lazy load model on first use
            if self.model is None:
                self.model = self._load_model(self.current_model_name)

            # Reload model if changed, using cache when possible
            if selected_model != self.current_model_name:
                self.current_model_name = selected_model
                self.model = self._load_model(selected_model)
                # Clear message history when model changes
                self.model.messages = []

            # Reload task if changed
            if selected_task != self.current_task_name:
                self.current_task_name = selected_task
                self.prompt_manager = self._load_task(selected_task)
                # Clear message history when task changes
                if self.model:
                    self.model.messages = []

            # Generate response using updated model & prompt manager
            prompt = self.prompt_manager.get_prompt(user_input=message)
            response_stream = self.model.generate_response(prompt)
            yield from response_stream

        chat_interface = gr.ChatInterface(
            chat_fn,
            textbox=gr.Textbox(placeholder="Ask me something...", container=False),
            additional_inputs=[
                gr.Dropdown(
                    choices=self.model_options,
                    value=self.current_model_name,
                    label="Select Gemma Model",
                ),
                gr.Dropdown(
                    choices=self.task_options,
                    value=self.current_task_name,
                    label="Select Task",
                ),
            ],
        )
        chat_interface.launch()

    def run(self):
        self._chat()

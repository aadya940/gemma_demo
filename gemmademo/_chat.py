import streamlit as st
from ._model import LlamaCppGemmaModel
from ._prompts import PromptManager


class StreamlitChat:
    """
    A class that handles the chat interface for the Gemma model.

    Features:
    - A Streamlit-based chatbot UI.
    -  Maintains chat history across reruns.
    -  Uses Gemma (Hugging Face) model for generating responses.
    -  Formats user inputs before sending them to the model.
    """

    def __init__(self, model: LlamaCppGemmaModel, prompt_manager: PromptManager):
        self.model = model
        self.prompt_manager = prompt_manager

    def run(self):
        self._chat()

    def _chat(self):
        st.title("Using model : " + self.model.get_model_name())
        self._build_states()

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            prompt = prompt.replace(
                "\n", "  \n"
            )  # Only double spaced backslash is rendered in streamlit for newlines.
            with st.chat_message("User"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "User", "content": prompt})

            prompt = self.prompt_manager.get_prompt(
                user_input=st.session_state.messages[-1]["content"]
            )
            response = self.model.generate_response(prompt).replace(
                "\n", "  \n"
            )  # Only double spaced backslash is rendered in streamlit for newlines.
            with st.chat_message("Gemma"):
                st.markdown(response)
            st.session_state.messages.append({"role": "Gemma", "content": response})

    def _build_states(self):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def clear_history(self):
        st.session_state.messages = []

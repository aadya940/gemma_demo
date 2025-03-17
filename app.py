# Interface all the functions from gemmademo.
# Implement a task selector in the side bar.
# Add a button to clear the chat history.
import streamlit as st
from gemmademo import (
    LlamaCppGemmaModel,
    StreamlitChat,
    PromptManager,
    huggingface_login,
)
import os
import sys
import subprocess


def main():
    # Page configuration
    st.set_page_config(page_title="Gemma Chat Demo", layout="wide")

    # Initialize session state variables
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemma-2b-it"
    if "selected_task" not in st.session_state:
        st.session_state.selected_task = "Question Answering"

    # Sidebar for login and configuration
    with st.sidebar:
        st.title("Gemma Chat Configuration")

        # Login section
        huggingface_login(os.getenv("HF_TOKEN"))
        # Model selection
        st.subheader("Model Selection")
        model_options = list(LlamaCppGemmaModel.AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Select Gemma Model",
            model_options,
            index=model_options.index(st.session_state.selected_model),
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.rerun()

        # Task selection
        st.subheader("Task Selection")
        task_options = ["Question Answering", "Text Generation", "Code Completion"]
        selected_task = st.selectbox(
            "Select Task",
            task_options,
            index=task_options.index(st.session_state.selected_task),
        )
        if selected_task != st.session_state.selected_task:
            st.session_state.selected_task = selected_task
            st.rerun()

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []  # Clear the chat history
            st.success("Chat history cleared!")

    # Main content area
    # Initialize model with the selected configuration
    model_name = st.session_state.selected_model
    model = LlamaCppGemmaModel(name=model_name)

    # Load model (will use cached version if available)
    with st.spinner(f"Loading {model_name}..."):
        model.load_model()

    # Initialize prompt manager with selected task
    prompt_manager = PromptManager(task=st.session_state.selected_task)

    # Initialize chat interface
    chat = StreamlitChat(model=model, prompt_manager=prompt_manager)
    st.session_state.chat_instance = chat

    # Run the chat interface
    chat.run()


if __name__ == "__main__":
    # Check if the script is being run directly with Python
    # If so, launch Streamlit programmatically
    if not os.environ.get("STREAMLIT_RUN_APP"):
        os.environ["STREAMLIT_RUN_APP"] = "1"
        # Get the current script path
        script_path = os.path.abspath(__file__)
        # Launch streamlit run with port 7860 and headless mode
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            script_path,
            "--server.port",
            "7860",
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ]
        subprocess.run(cmd)
    else:
        # Normal Streamlit execution
        main()

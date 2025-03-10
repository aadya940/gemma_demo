# Interface all the functions from gemmademo.
# Implement login functionality in the side bar.
# Implement a task selector in the side bar.
# Interface all the functions from gemmademo.
# Add a button to clear the chat history.

import streamlit as st
from gemmademo import HuggingFaceGemmaModel, StreamlitChat, PromptManager, huggingface_login
import os
import sys
import subprocess

def main():
    # Page configuration
    st.set_page_config(page_title="Gemma Chat Demo", layout="wide")

    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemma-2b-it"
    if "selected_task" not in st.session_state:
        st.session_state.selected_task = "Question Answering"

    # Sidebar for login and configuration
    with st.sidebar:
        st.title("Gemma Chat Configuration")
        
        # Login section
        st.subheader("Login")
        if not st.session_state.authenticated:
            hf_token = st.text_input("Hugging Face Token", type="password")
            if st.button("Login"):
                try:
                    huggingface_login(hf_token)
                    st.session_state.authenticated = True
                    st.success("Successfully logged in!")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
        else:
            st.success("Logged in to Hugging Face")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.rerun()
        
        # Model selection
        st.subheader("Model Selection")
        model_options = list(HuggingFaceGemmaModel.AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Select Gemma Model",
            model_options,
            index=model_options.index(st.session_state.selected_model)
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
            index=task_options.index(st.session_state.selected_task)
        )
        if selected_task != st.session_state.selected_task:
            st.session_state.selected_task = selected_task
            st.rerun()
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            if "chat_instance" in st.session_state:
                st.session_state.chat_instance.clear_history()
            st.rerun()

    # Main content area
    if st.session_state.authenticated:
        # Initialize model with the selected configuration
        model_name = HuggingFaceGemmaModel.AVAILABLE_MODELS[st.session_state.selected_model]["name"]
        model = HuggingFaceGemmaModel(name=model_name)
        
        # Load model (will use cached version if available)
        with st.spinner(f"Loading {model_name}..."):
            model.load_model(device_map="auto")
        
        # Initialize prompt manager with selected task
        prompt_manager = PromptManager(task=st.session_state.selected_task)
        
        # Initialize chat interface
        chat = StreamlitChat(model=model, prompt_manager=prompt_manager)
        st.session_state.chat_instance = chat
        
        # Run the chat interface
        chat.run()
    else:
        st.info("Please login with your Hugging Face token in the sidebar to start chatting.")

if __name__ == "__main__":
    # Check if the script is being run directly with Python
    # If so, launch Streamlit programmatically
    if not os.environ.get('STREAMLIT_RUN_APP'):
        os.environ['STREAMLIT_RUN_APP'] = '1'
        # Get the current script path
        script_path = os.path.abspath(__file__)
        # Launch streamlit run with port 7860 and headless mode
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, 
               "--server.port", "7860",
               "--server.address", "0.0.0.0",
               "--server.headless", "true"]
        subprocess.run(cmd)
    else:
        # Normal Streamlit execution
        main()

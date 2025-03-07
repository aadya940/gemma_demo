from gemma_demo import StreamlitChat, HuggingFaceGemmaModel, PromptManager
import streamlit as st
import os
import sys
import subprocess
from huggingface_hub import login

def main():
    # Set page config must be the first Streamlit command
    st.set_page_config(
        page_title="Gemma Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Gemma 2B Chat Interface")
    
    # Login section
    with st.sidebar:
        st.header("Hugging Face Authentication")
        
        # Token input field - only user input, no reading from file
        token = st.text_input(
            "Enter your Hugging Face token:",
            type="password",
            help="Get your token from huggingface.co/settings/tokens"
        )
        
        # Login button
        if st.button("Login"):
            if token:
                try:
                    # Authenticate with Hugging Face
                    login(token)
                    st.success("Successfully logged in to Hugging Face!")
                    
                    # Store token in session state for this session
                    st.session_state['authenticated'] = True
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
                    st.session_state['authenticated'] = False
            else:
                st.error("Please enter a valid token")
                st.session_state['authenticated'] = False
    
    # Initialize 'authenticated' in session state if it doesn't exist
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    # Only load model and render chat if authenticated
    if st.session_state['authenticated']:
        # Initialize the model with explicit device mapping
        with st.spinner("Loading model... This may take a moment."):
            model = HuggingFaceGemmaModel("google/gemma-2b").load_model(device_map="cpu")
            
            # Initialize the prompt manager
            prompt_manager = PromptManager(task="question_answering")
            
            # Initialize and launch the chat interface
            chat = StreamlitChat(model, prompt_manager)
            chat.render()
    else:
        st.info("Please login with your Hugging Face token to start chatting.")

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

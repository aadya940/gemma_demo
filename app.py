from gemma_demo import StreamlitChat, HuggingFaceGemmaModel, PromptManager
import streamlit as st
import os
import sys
import subprocess
from huggingface_hub import login

def main():
    # Set page config must be the first Streamlit command
    st.set_page_config(
        page_title="Gemma AI Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://huggingface.co/google/gemma-2b',
            'Report a bug': 'https://github.com/google/gemma',
            'About': 'This is a demo of Google\'s Gemma language models using Streamlit.'
        }
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Input field styling */
        .stTextInput input {
            border-radius: 6px;
        }
        
        /* Chat message styling */
        .stChatMessage {
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Chat input styling */
        .stChatInputContainer {
            padding-top: 1rem;
            border-top: 1px solid #f0f0f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemma_logo.max-600x600.png", width=80)
    with col2:
        st.title("Gemma AI Chat")
        st.markdown("*Powered by Google's Gemma language models*")
    
    st.markdown("---")
    
    # Login section
    with st.sidebar:
        st.header("🔑 Authentication")
        
        # Token input field - only user input, no reading from file
        token = st.text_input(
            "Enter your Hugging Face token:",
            type="password",
            help="Get your token from huggingface.co/settings/tokens"
        )
        
        # Login button with improved styling
        if st.button("🔐 Login to Hugging Face", type="primary", use_container_width=True):
            if token:
                with st.spinner("Authenticating..."):
                    try:
                        # Authenticate with Hugging Face
                        login(token)
                        st.success("✅ Successfully logged in to Hugging Face!")
                        
                        # Store token in session state for this session
                        st.session_state['authenticated'] = True
                    except Exception as e:
                        st.error(f"❌ Login failed: {str(e)}")
                        st.session_state['authenticated'] = False
            else:
                st.error("⚠️ Please enter a valid token")
                st.session_state['authenticated'] = False
        
        st.markdown("---")
        
        # Add helpful information
        with st.expander("ℹ️ About Gemma Models"):
            st.markdown("""
            **Gemma** is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.
            
            These models are designed to be:
            - **Efficient**: Optimized for various devices
            - **Helpful**: Trained on diverse datasets
            - **Safe**: Developed with Google's AI Principles
            
            [Learn more about Gemma](https://blog.google/technology/ai/google-gemma-open-model/)
            """)
    
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
        # Show welcome message and login instructions
        st.info("👋 Welcome to Gemma AI Chat!")
        
        # Create a card-like container for login instructions
        with st.container():
            st.markdown("""
            <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin: 20px 0;">
                <h3>Getting Started</h3>
                <p>To start chatting with Gemma, you'll need to authenticate with your Hugging Face account:</p>
                <ol>
                    <li>Create a Hugging Face account if you don't have one</li>
                    <li>Get your access token from <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a></li>
                    <li>Enter your token in the sidebar and click "Login"</li>
                </ol>
                <p><strong>Note:</strong> You need to have accepted the <a href="https://huggingface.co/google/gemma-2b" target="_blank">Gemma model license</a> on Hugging Face.</p>
            </div>
            """, unsafe_allow_html=True)
            
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

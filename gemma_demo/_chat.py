import streamlit as st
import time
import random

class StreamlitChat:
    """
    Implements a chat interface using Streamlit between the user and the model.
    Also interprets markdown responses from the model and renders them in the chat interface.
    Handles different tasks in the chat interface.
    """
    def __init__(self, model, prompt_manager):
        self.model = model
        self.prompt_manager = prompt_manager
        self.tasks = {
            "Question Answering": "question_answering",
            "Text Generation": "text_generation",
            "Code Completion": "code_completion"
        }
        self.available_models = list(model.AVAILABLE_MODELS.keys())
        self.current_model_key = "gemma-2b" if model.name == "google/gemma-2b" else "gemma-2b-it"
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Add a welcome message if chat is empty
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "👋 Hello! I'm Gemma, an AI assistant powered by Google's language models. How can I help you today?"
            })
    
    def chat(self, message, task):
        """Process a chat message and update history"""
        # Update the prompt manager's task
        self.prompt_manager.task = self.tasks[task]
        
        # Format the prompt based on the task
        if self.prompt_manager.task == "question_answering":
            prompt = self.prompt_manager.get_prompt(question=message)
        elif self.prompt_manager.task == "text_generation":
            prompt = self.prompt_manager.get_prompt(topic=message)
        elif self.prompt_manager.task == "code_completion":
            prompt = self.prompt_manager.get_prompt(code=message)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Create a placeholder for the assistant's message
        with st.chat_message("assistant", avatar="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemma_logo.max-600x600.png"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show a typing indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown("*Thinking...*")
            
            # Generate response
            response = self.model.generate_response(prompt)
            
            # Special handling for code completion
            if self.prompt_manager.task == "code_completion" and "```" not in response:
                response = self.model.generate_response(prompt + "\nPlease format your response as a proper code block with the correct language identifier.")
            
            # Simulate typing effect for a more natural feel
            typing_placeholder.empty()
            for i in range(len(response)):
                full_response = response[:i+1]
                message_placeholder.markdown(full_response)
                if i % 3 == 0:  # Only sleep occasionally to speed up the effect
                    time.sleep(0.01)
            
            # Display the final response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def change_task(self, task):
        """Change the current task and return its description"""
        task_descriptions = {
            "Question Answering": "Ask any factual question or seek explanations on various topics.",
            "Text Generation": "Generate creative writing, stories, or content on a given topic.",
            "Code Completion": "Complete or improve code snippets in various programming languages."
        }
        
        return task_descriptions.get(task, "")
    
    def change_model(self, model_key):
        """Change the current model"""
        try:
            self.current_model_key = model_key
            model_info = self.model.AVAILABLE_MODELS[model_key]
            self.model = HuggingFaceGemmaModel(model_info["name"]).load_model()
            return f"Successfully loaded {model_info['name']}"
        except Exception as e:
            return f"Failed to load model: {str(e)}"
    
    def render(self):
        """Render the Streamlit interface"""        
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
            /* Chat container styling */
            .chat-container {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                margin-bottom: 20px;
                max-height: 800px;
                overflow-y: auto;
            }
            
            /* Message styling */
            .stChatMessage {
                background-color: transparent !important;
                border: none !important;
                padding: 1rem 0 !important;
            }
            
            /* User message styling */
            .stChatMessage [data-testid="chatAvatarIcon-user"] {
                background-color: #4285F4 !important;
            }
            
            /* Assistant message styling */
            .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
                background-color: #fbbc05 !important;
            }
            
            /* Message content styling */
            .stChatMessage > div {
                background-color: #f8f9fa !important;
                border-radius: 12px !important;
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
            }
            
            /* Code block styling */
            .stMarkdown pre {
                background-color: #1e1e1e !important;
                border-radius: 8px !important;
                padding: 12px !important;
                margin: 10px 0 !important;
                overflow-x: auto !important;
                border: 1px solid #3e3e3e !important;
            }
            
            .stMarkdown code {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
                font-size: 14px !important;
                line-height: 1.5 !important;
                color: #e6e6e6 !important;  /* Light gray text for code */
            }
            
            /* Inline code styling */
            .stMarkdown p code {
                background-color: #f0f0f0 !important;
                color: #24292e !important;
                padding: 2px 6px !important;
                border-radius: 4px !important;
            }
            
            /* Syntax highlighting for code blocks */
            .stMarkdown pre code {
                color: #e6e6e6 !important;  /* Base text color */
            }
            .stMarkdown pre code .keyword { color: #569cd6 !important; }  /* Keywords */
            .stMarkdown pre code .string { color: #ce9178 !important; }   /* Strings */
            .stMarkdown pre code .comment { color: #6a9955 !important; }  /* Comments */
            .stMarkdown pre code .function { color: #dcdcaa !important; } /* Functions */
            .stMarkdown pre code .number { color: #b5cea8 !important; }   /* Numbers */
            
            /* Task card styling */
            .task-card {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #4285F4;
            }
            
            /* Model card styling */
            .model-card {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #34A853;
            }
            
            /* Chat input styling */
            .stChatInputContainer {
                padding-top: 1rem;
                border-top: 1px solid #f0f0f0;
                position: sticky;
                bottom: 0;
                background-color: white;
            }
            
            /* Streamlit elements styling */
            .stTextInput > div > div {
                padding: 0.5rem 0.75rem;
            }
            
            .stButton button {
                width: 100%;
                padding: 0.5rem;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                transition: all 0.2s ease;
            }
            
            .stButton button:hover {
                background-color: #e9ecef;
                border-color: #dee2e6;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Main layout with improved proportions
        col1, col2 = st.columns([7, 3])
        
        # Main chat area
        with col1:
            # Chat container with fixed height and scrolling
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat messages with avatars
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemma_logo.max-600x600.png"):
                        st.markdown(message["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input (moved outside the scrollable container)
            task_placeholders = {
                "Question Answering": "Ask me anything...",
                "Text Generation": "Describe what you'd like me to write about...",
                "Code Completion": "Enter code or describe what you need..."
            }
            
            current_task = list(self.tasks.keys())[0]  # Default
            for task_name, task_id in self.tasks.items():
                if task_id == self.prompt_manager.task:
                    current_task = task_name
                    break
            
            placeholder_text = task_placeholders.get(current_task, "Type your message here...")
            
            # Chat input
            if user_input := st.chat_input(placeholder_text):
                self.chat(user_input, current_task)
        
        # Sidebar with controls
        with col2:
            # Model selection card
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 Model")
            
            model_dropdown = st.selectbox(
                "Select Model",
                options=self.available_models,
                index=self.available_models.index(self.current_model_key),
                format_func=lambda x: f"{x} ({self.model.AVAILABLE_MODELS[x]['description']})"
            )
            
            # Model info
            st.markdown(f"""
            **Current Model:** {self.model.AVAILABLE_MODELS[self.current_model_key]['name']}  
            **Type:** {self.model.AVAILABLE_MODELS[self.current_model_key]['type']}  
            **Status:** Active and ready
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Task selection card
            st.markdown('<div class="task-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Task")
            
            task_dropdown = st.selectbox(
                "Select Task",
                options=list(self.tasks.keys()),
                index=0
            )
            
            task_description = self.change_task(task_dropdown)
            st.markdown(f"**Description:** {task_description}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generation settings
            with st.expander("⚙️ Generation Settings", expanded=False):
                st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1, 
                         help="Higher values make output more random, lower values more deterministic")
                st.slider("Max Length", min_value=64, max_value=1024, value=512, step=64,
                         help="Maximum number of tokens to generate")
                st.checkbox("Stream Output", value=True, 
                           help="Show the response as it's being generated")
            
            # Examples section
            with st.expander("💡 Example Prompts", expanded=False):
                examples = {
                    "Question Answering": [
                        "Explain how neural networks work",
                        "What is quantum computing?",
                        "How does photosynthesis work?"
                    ],
                    "Text Generation": [
                        "Write a short story about space exploration",
                        "Create a poem about nature",
                        "Write a product description for a smart watch"
                    ],
                    "Code Completion": [
                        "Write a Python function to sort a list",
                        "Create a React component for a login form",
                        "Write a SQL query to find the top 5 customers"
                    ]
                }
                
                current_examples = examples.get(task_dropdown, [])
                for example in current_examples:
                    if st.button(example, key=f"example_{example}", use_container_width=True):
                        self.chat(example, task_dropdown)
            
            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                    # Add a welcome message when clearing chat
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": "👋 Hello! I'm Gemma, an AI assistant powered by Google's language models. How can I help you today?"
                    }]
                    st.experimental_rerun()
            
            with col_b:
                if st.button("📋 Copy Last", type="secondary", use_container_width=True):
                    if len(st.session_state.messages) > 0:
                        last_message = st.session_state.messages[-1]["content"]
                        st.code(last_message)
                        st.toast("Response copied to clipboard!")
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.8em;">
                Powered by Google's Gemma models<br>
                <a href="https://huggingface.co/google/gemma-2b" target="_blank">Model Documentation</a> | 
                <a href="https://github.com/google/gemma" target="_blank">GitHub</a>
            </div>
            """, unsafe_allow_html=True)
        
        # Handle model change
        if model_dropdown != self.current_model_key:
            with st.spinner(f"Loading {model_dropdown}..."):
                result = self.change_model(model_dropdown)
                st.toast(result)
                
from gemma_demo._model import HuggingFaceGemmaModel

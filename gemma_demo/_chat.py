import streamlit as st
import time

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
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Generate response
            response = self.model.generate_response(prompt)
            
            # Special handling for code completion
            if self.prompt_manager.task == "code_completion" and "```" not in response:
                response = self.chat(message, task)
            
            # Display the final response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def change_task(self, task):
        """Change the current task"""
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
        st.set_page_config(
            page_title="Gemma Chat",
            page_icon="🤖",
            layout="wide"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main {
                max-width: 1200px;
                margin: auto;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
            }
            .header h1 {
                margin-bottom: 5px;
            }
            .header p {
                margin-top: 0;
                color: #666;
            }
            .model-info {
                padding: 10px;
                border-radius: 8px;
                background: #f0f4ff;
                margin-bottom: 10px;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                font-size: 0.8em;
                color: #666;
            }
            
            /* Code block styling */
            pre {
                background-color: #1e1e1e !important;
                border-radius: 8px !important;
                padding: 12px !important;
                margin: 10px 0 !important;
                overflow-x: auto !important;
                border: 1px solid #3e3e3e !important;
            }
            
            code {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
                font-size: 14px !important;
                line-height: 1.5 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="header">
            <h1>Gemma Chat Interface</h1>
            <p>Interact with Google's Gemma language models through a simple chat interface</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main layout
        col1, col2 = st.columns([7, 3])
        
        # Sidebar with controls
        with col2:
            with st.container():
                st.markdown("## Model Settings")
                
                with st.container():
                    st.markdown('<div class="model-info">', unsafe_allow_html=True)
                    model_dropdown = st.selectbox(
                        "Select Model",
                        options=self.available_models,
                        index=self.available_models.index(self.current_model_key)
                    )
                    
                    model_status = st.markdown(f"""
                    ### Model: {self.model.AVAILABLE_MODELS[self.current_model_key]['description']}
                    
                    Currently loaded and ready
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("## Task Settings")
                
                task_dropdown = st.selectbox(
                    "Select Task",
                    options=list(self.tasks.keys()),
                    index=0
                )
                
                task_description = self.change_task(task_dropdown)
                st.markdown(f"""
                ### Current Task: {task_dropdown}
                
                {task_description}
                """)
                
                with st.expander("Help & Examples", expanded=False):
                    st.markdown("""
                    ### Example Prompts
                    
                    **Question Answering**:
                    - "What is quantum computing?"
                    - "Explain how neural networks work"
                    
                    **Text Generation**:
                    - "Write a short story about space exploration"
                    - "Create a poem about nature"
                    
                    **Code Completion**:
                    - "Write a Python function to sort a list"
                    - ```python
                      def fibonacci(n):
                          # Complete this function
                      ```
                    """)
                
                if st.button("Clear Chat", type="secondary"):
                    st.session_state.messages = []
                    st.experimental_rerun()
            
        # Main chat area
        with col1:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if user_input := st.chat_input("Type your message here..."):
                self.chat(user_input, task_dropdown)
                
        # Footer
        st.markdown("""
        <div class="footer">
            Powered by Google's Gemma models via Hugging Face | Created with Streamlit
        </div>
        """, unsafe_allow_html=True)
        
        # Handle model change
        if model_dropdown != self.current_model_key:
            with st.spinner(f"Loading {model_dropdown}..."):
                result = self.change_model(model_dropdown)
                st.toast(result)


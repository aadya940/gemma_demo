from gemma_demo import StreamlitChat, HuggingFaceGemmaModel, PromptManager

def main():
    # Initialize the model with explicit device mapping
    model = HuggingFaceGemmaModel("google/gemma-2b").load_model(device_map="cpu")
    
    # Initialize the prompt manager
    prompt_manager = PromptManager(task="question_answering")
    
    # Initialize and launch the chat interface
    chat = StreamlitChat(model, prompt_manager)
    chat.render()

if __name__ == "__main__":
    main()

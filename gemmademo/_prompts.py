class PromptManager:
    """
    A manager for generating prompts based on different tasks.

    This class provides methods to format user input into prompts suitable for
    various tasks such as Question Answering, Text Generation, and Code Completion.
    It raises a ValueError if an unsupported task is specified.
    """
    def __init__(self, task):
        self.task = task

    def get_prompt(self, user_input):
        if self.task == "Question Answering":
            return self.get_question_answering_prompt(user_input)
        elif self.task == "Text Generation":
            return self.get_text_generation_prompt(user_input)
        elif self.task == "Code Completion":
            return self.get_code_completion_prompt(user_input)
        else:
            raise ValueError(f"Task {self.task} not supported")

    def get_question_answering_prompt(self, user_input):
        """
        Format user input for question answering task
        """
        prompt = f"""You are a helpful AI assistant. Answer the following question accurately and concisely.
        Only answer the question, do not provide any other information.

        Question: {user_input}

        Answer:"""
        return prompt

    def get_text_generation_prompt(self, user_input):
        """
        Format user input for text generation task
        """
        prompt = f"""Continue the following text in a coherent and engaging way:
        Only continue the text, do not provide any other information.
        
        {user_input}

        Continuation:"""
        return prompt

    def get_code_completion_prompt(self, user_input):
        """
        Format user input for code completion task
        """
        prompt = f"""Complete the following code snippet directly with proper syntax and 
        without explanations or extra text:
        {user_input}"""
        return prompt

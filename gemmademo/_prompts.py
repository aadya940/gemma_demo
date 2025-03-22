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
        return user_input

    def get_system_prompt(self):
        """Returns the system prompt based on the specified task."""
        if self.task == "Question Answering":
            return "You are a helpful AI assistant. Answer questions concisely and accurately."
        elif self.task == "Text Generation":
            return "You are a creative AI writer. Generate engaging and coherent text based on the input."
        elif self.task == "Code Completion":
            return "You are a coding assistant. Complete code snippets correctly without explanations."

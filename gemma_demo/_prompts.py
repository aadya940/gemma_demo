class PromptManager:
    """
    A class for managing prompts based on the task.
    Supported tasks:
        - Text Generation
        - Code Completion
        - Question Answering
    
    Contains prompt templates for each task.
    """
    def __init__(self, task="question_answering"):
        self.task = task
        self.templates = {
            "text_generation": "Write a creative text about {topic}:\n\n",
            "code_completion": (
                "Complete or improve the following code. Format your response as a proper markdown "
                "code block with the correct language identifier.\n\n"
                "Example format:\n"
                "```python\n"
                "def example():\n"
                "    pass\n"
                "```\n\n"
                "Here's the code to complete:\n"
                "{code}\n"
            ),
            "question_answering": "Answer the following question:\n{question}\n"
        }
        
    def get_prompt(self, **kwargs):
        """
        Get a formatted prompt based on the task and provided parameters
        
        Parameters:
        -----------
        **kwargs: dict
            Parameters to format the prompt template with
        
        Returns:
        --------
        str: The formatted prompt
        """
        if self.task not in self.templates:
            raise ValueError(f"Task {self.task} not supported")
            
        return self.templates[self.task].format(**kwargs)

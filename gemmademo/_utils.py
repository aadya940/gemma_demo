def huggingface_login(token: str):
    """
    Login to Hugging Face using the token
    """
    from huggingface_hub import login
    login(token=token)

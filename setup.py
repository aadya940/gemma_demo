from setuptools import setup, find_packages

setup(
    name="gemma_demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "gradio>=4.0.0",
        "huggingface_hub>=0.19.0",
        "markdown>=3.4.0",
    ],
    description="A demo application for the Gemma model from Google using Hugging Face",
    python_requires=">=3.8",
)

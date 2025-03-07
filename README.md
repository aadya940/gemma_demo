---
title: Gemma Chat Interface
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
python_version: 3.12
app_file: app.py
pinned: false
suggested_hardware: t4-small
models:
  - google/gemma-2b
  - google/gemma-2b-it
  - google/gemma-7b
  - google/gemma-7b-it
tags:
  - gemma
  - chat
  - language-model
  - code-generation
short_description: A chat interface for Google's Gemma models.
---

# Gemma Demo

A lightweight chat interface for interacting with Google's Gemma language models via Hugging Face.

## Features

- 💬 **Interactive Chat Interface** built with Gradio
- 🔄 **Multiple Task Modes**:
  - Question Answering
  - Text Generation
  - Code Completion
- 📝 **Markdown Support** for rich text responses
- 🧩 **Modular Architecture** for easy extension

## Quick Start

### Prerequisites

- Python 3.8+
- Hugging Face account with Gemma model access

### Installation

```bash
# Clone repository (if applicable)
git clone https://github.com/aadya940/gemma_demo.git
cd gemma_demo

# Install dependencies
pip install -r requirements.txt
pip install .

# Log in to Hugging Face
huggingface-cli login
```

### Running the Demo

```bash
python -m gemma_demo
```

## Usage Guide

1. **Select a Task** from the dropdown menu
2. **Enter your prompt** in the text box
3. **View the model's response** in the chat window

### Example Prompts

- **Question Answering**: "What are the main applications of machine learning?"
- **Text Generation**: "Write a short story about a robot learning to paint"
- **Code Completion**: "Write a Python function to calculate Fibonacci numbers"

## Project Structure

```
gemma_demo/
├── __init__.py    # Package initialization
├── __main__.py    # Entry point
├── _chat.py       # Gradio interface
├── _model.py      # Gemma model wrapper
└── _prompts.py    # Prompt templates
```

## Requirements

- transformers >= 4.35.0
- torch >= 2.0.0
- gradio >= 4.0.0
- huggingface_hub >= 0.19.0
- markdown >= 3.4.0

## Important Note

⚠️ **Hugging Face Login Required**: You must have access to Gemma models. Visit [huggingface.co](https://huggingface.co/welcome) to create an account and request access.
---
title: Gemma Chat Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
hardware: true
---

# Gemma Chat Demo

An interactive chat application powered by Google's Gemma models using Hugging Face and Streamlit.

## Features

- 🔐 Hugging Face authentication for accessing Gemma models
- 🤖 Support for multiple Gemma model variants (2B, 7B, base and instruction-tuned)
- 🔄 Task selection for different conversation types:
  - Question Answering
  - Text Generation
  - Code Completion
- 💬 Clean chat interface with message history
- 🧹 Option to clear chat history

## Usage

1. Log in with your Hugging Face token in the sidebar
2. Select your preferred Gemma model
3. Choose a task type for your conversation
4. Start chatting with the model!

## Requirements

- Python 3.8+
- Hugging Face account with access to Gemma models
- Dependencies listed in requirements.txt

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
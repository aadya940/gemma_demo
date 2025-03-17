---
title: Gemma Chat Interface
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.43.1
python_version: 3.12
app_file: app.py
pinned: false
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

# Gemma Chat Demo

An interactive chat application powered by Google's Gemma models using Hugging Face and Streamlit.

HuggingFace Spaces Link: https://huggingface.co/spaces/aadya1762/GemmaDemoSt2

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

## Limitations
- CPU Implementation is very slow (a simple code completion can take around 10 minutes).
- Limit Scope of Optimization (`torch.compile` takes several minutes to compile & recompiles frequently)

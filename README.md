---
title: Gemma Chat Interface
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: 3.12
app_file: app.py
pinned: false
models:
  - google/gemma-2b
  - google/gemma-3b
  - google/gemma-7b
tags:
  - gemma
  - chat
  - language-model
  - code-generation
short_description: A friendly chat interface for Google's Gemma models.
---

# Gemma Chat Demo

A simple and friendly chat application powered by Google's Gemma AI models.

## Features

- 💬 Chat with Google's Gemma AI models right on your computer
- 🔄 Choose between different models:
  - gemma-3b: Balanced performance (recommended)
  - gemma-2b: Faster responses
  - gemma-7b: More capable but requires more resources
- 🧩 Different chat modes:
  - Question Answering: Get helpful answers to your questions
  - Text Generation: Continue any text or story
  - Code Completion: Get help with coding
- 🖥️ Simple, easy-to-use interface
- 📚 No complex setup needed

## Getting Started

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. When the application starts:
   - Select a model from the dropdown
   - Choose a chat mode
   - Start typing and chatting!

## Requirements

- Python 3.11 or newer 
- Internet connection for first-time model download

## Tips

- For code completion, start by typing the beginning of your code
- The first time you select a model, it may take a moment to download
- You can switch between models anytime during your conversation

Enjoy chatting with Gemma!

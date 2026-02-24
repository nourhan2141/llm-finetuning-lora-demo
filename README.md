LLM Fine-Tuning with LoRA + Streamlit Demo
Overview

This project demonstrates parameter-efficient fine-tuning of a decoder-only language model using LoRA (Low-Rank Adaptation), followed by deployment as an interactive chatbot using Streamlit.

The goal was to understand:

How LoRA reduces trainable parameters

How fine-tuning affects response behavior

How to turn a trained model into a simple web-based interface

This is a learning/demo project.

Tech Stack

Python

PyTorch

Hugging Face Transformers

PEFT (LoRA)

Streamlit

Model & Training

Base model: (add your base model name here, e.g., GPT-Neo 125M)

Fine-tuning method: LoRA via PEFT

Training setup: Parameter-efficient fine-tuning (<1% trainable parameters)

Objective: Improve response style/behavior on a specific dataset

Training was done in a Jupyter notebook (training_notebook.ipynb).

Key steps:

Load pretrained model from Hugging Face

Apply LoRA adapters

Fine-tune on custom dataset

Save trained adapter weights

Run inference tests

Streamlit Demo App

The fine-tuned model is wrapped in a simple chatbot interface built with Streamlit.

Features:

User text input

Model-generated response

Basic prompt formatting

Lightweight demo interface

Run locally:

pip install -r requirements.txt
streamlit run app.py
Example Demo

The demo video includes 3 test queries to evaluate:

Standard prompt

Structured question

Slightly ambiguous or edge-case input

This helps observe how fine-tuning changes response consistency and tone.

Project Structure
├── training_notebook.ipynb
├── app.py
├── requirements.txt
├── README.md
What I Learned

LoRA makes fine-tuning feasible on limited hardware.

Prompt structure strongly influences output quality.

Turning a model into an interface reveals usability issues that are not obvious during notebook testing.

Inference speed and response formatting matter for user experience.

Limitations

This is a demo-level chatbot, not production-ready.

No advanced guardrails or moderation layer.

Limited evaluation and benchmarking.

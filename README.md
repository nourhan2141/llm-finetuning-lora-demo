# LLM Fine-Tuning with LoRA + Streamlit Demo

## Overview

This project demonstrates parameter-efficient fine-tuning of a decoder-only language model using LoRA (Low-Rank Adaptation) and deployment as an interactive chatbot using Streamlit.

## Objectives

* Reduce trainable parameters using LoRA
* Analyze how fine-tuning alters response behavior
* Convert a trained model into a simple web-based interface

This is a learning-focused implementation and is not production-ready.
## Model & Training
Base Model: Add model name here
Fine-Tuning Method: LoRA via PEFT
Trainable Parameters: < 1% of total parameters

Training was conducted in:
```
FineTuningTask.ipynb
```
## Training Workflow
1. Load pretrained model
2. Attach LoRA adapters
3. Fine-tune on custom dataset
4. Save adapter weights
5. Run inference validation

## Streamlit Demo Application

The fine-tuned model is wrapped in a lightweight chatbot interface.

* Features
* Text input
* Model-generated responses
* Prompt formatting

## Demo Evaluation

### Three test queries were used:
* Standard prompt
* Structured question
* Ambiguous or edge-case input

## Key Observations

* LoRA enables efficient fine-tuning on limited hardware.
* Prompt structure significantly influences output quality.
* Deployment reveals usability issues not visible during notebook testing.
* Inference latency impacts user experience
These demonstrate changes in response consistency and tone after fine-tuning.
Simple interactive layout

## Limitations

* Demo-level chatbot
* No advanced safety or moderation layer
* Limited quantitative evaluation

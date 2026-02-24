import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
MODEL_NAME = "nourhan214/gptneo125m-bitext-customer-support"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype
    )
    model.eval()
    model.to(device)
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# HELPER FUNCTION 
def stream_text(text, delay=0.03):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# UI SETUP
st.title("Customer Support Assistant")
st.caption("Powered by LoRA Fine-tuned GPT-Neo 125M")

#  CHAT LOGIC
if user_input := st.chat_input("Ask a question..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Format prompt (Single-turn only)
    prompt = f"User: {user_input}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  
                    temperature=0.3,           
                    top_p=0.9,
                    repetition_penalty=1.15,   
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_reply = full_response.split("Assistant:")[-1].strip()
            
        # 4. Stream the output to the UI
        st.write_stream(stream_text(bot_reply))
import streamlit as st
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpt'))
from model import BigramLanguageModel

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'loop', 'checkpoints', 'model.pt')
MAX_NEW_TOKENS  = 200
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    chars      = checkpoint['chars']
    vocab_size = checkpoint['vocab_size']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, stoi, itos

def encode(s, stoi):
    return [stoi.get(c, 0) for c in s]

def decode(l, itos):
    return ''.join([itos.get(i, '') for i in l])

def run_generate(prompt, model, stoi, itos):
    context = torch.tensor(encode(prompt, stoi), dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)
    return decode(out[0].tolist(), itos)

# ── Page config ──
st.set_page_config(page_title="Mini GPT-2", page_icon="", layout="centered")

# ── Load model ──
with st.spinner("Loading model…"):
    model, stoi, itos = load_model()

if model is None:
    st.warning("⏳ Model not loaded yet. Place your checkpoint at the correct path then refresh.")
    st.stop()

# ── Header ──
st.title(" Mini GPT-2")
st.caption("Decoder-only transformer trained from scratch · WikiText-103 · Karpathy-style")
params = sum(p.numel() for p in model.parameters()) / 1e6
st.success(f" Model loaded — **{params:.2f}M parameters** · **{DEVICE.upper()}**")
st.divider()

# ── Chat ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Say something…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            full_output = run_generate(prompt, model, stoi, itos)
            response    = full_output[len(prompt):]
        st.markdown(response if response.strip() else "_[no output — try a longer prompt]_")
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.messages:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

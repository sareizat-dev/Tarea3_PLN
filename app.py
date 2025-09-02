import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="El Oráculo del Quijote", page_icon="📜", layout="wide")
st.title("📜 El Oráculo del Quijote 🖋️")
st.write("Pregúntale al modelo fine-tuneado sobre Don Quijote y su mundo.")

# ---------------------------
# Cargar modelo en 4-bit
# ---------------------------
@st.cache_resource
def cargar_modelo(hf_token=None):
    model_id = "sareizat-dev/qwen-quijote-merged"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=None,        # fuerza CPU
        low_cpu_mem_usage=True  # ayuda en RAM
        token=hf_token,
        trust_remote_code=True
    )
    return model, tokenizer

# ---------------------------
# Token
# ---------------------------
hf_token = st.secrets.get("HF_TOKEN", None)

# ---------------------------
# Interfaz principal
# ---------------------------
if hf_token:
    model, tokenizer = cargar_modelo(hf_token)

    pregunta = st.text_area("Tu pregunta:", "¿Quién es el escudero de Don Quijote?")
    if st.button("Obtener respuesta"):
        if pregunta.strip():
            prompt = f"""
<|im_start|>system
Eres un erudito experto en la obra 'Don Quijote de la Mancha'. Responde con tono clásico y preciso.
<|im_end|>
<|im_start|>user
{pregunta}
<|im_end|>
<|im_start|>assistant
"""

            with st.spinner("Pensando..."):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extraer respuesta del asistente
            start_token = "<|im_start|>assistant\n"
            start_index = full_response.rfind(start_token)
            if start_index != -1:
                respuesta = full_response[start_index + len(start_token):].strip()
                end_token = "<|im_end|>"
                if end_token in respuesta:
                    respuesta = respuesta.split(end_token)[0].strip()
            else:
                respuesta = full_response

            st.success("Respuesta del Oráculo:")
            st.write(respuesta)
else:
    st.warning("⚠️ Agrega tu `HF_TOKEN` en los Secrets de Streamlit Cloud.")

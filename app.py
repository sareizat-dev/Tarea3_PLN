import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title="📜 El Oráculo del Quijote", page_icon="🖋️")

st.title("📜 El Oráculo del Quijote 🖋️")
st.write("Haz preguntas sobre Don Quijote y el modelo responderá en estilo cervantino.")

# ---------------------------
# Cargar modelo y tokenizer
# ---------------------------
@st.cache_resource
def cargar_modelo():
    try:
        model_id = "sareizat-dev/qwen-quijote-merged"

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,          # Fuerza CPU en Streamlit Cloud
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        return model, tokenizer
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

model, tokenizer = cargar_modelo()

# ---------------------------
# Lógica principal
# ---------------------------
if model and tokenizer:
    pregunta = st.text_area(
        "Ingresa tu pregunta:",
        "¿Quién es el escudero de Don Quijote?"
    )

    if st.button("Obtener respuesta"):
        if pregunta.strip():
            prompt_chatml = f"""
<|im_start|>system
Eres un erudito experto en la obra "Don Quijote de la Mancha" de Miguel de Cervantes. 
Responde de forma precisa y con tono cervantino.
<|im_end|>
<|im_start|>user
{pregunta}
<|im_end|>
<|im_start|>assistant
"""

            with st.spinner("Pensando en la respuesta..."):
                inputs = tokenizer(prompt_chatml, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
            start_token = "<|im_start|>assistant"
            start_index = decoded.rfind(start_token)
            if start_index != -1:
                respuesta = decoded[start_index + len(start_token):].strip()
                end_token = "<|im_end|>"
                if end_token in respuesta:
                    respuesta = respuesta.split(end_token)[0].strip()

                st.success("Respuesta del Oráculo:")
                st.write(respuesta)
            else:
                st.warning("No se pudo generar una respuesta clara.")
        else:
            st.warning("Por favor, escribe una pregunta.")

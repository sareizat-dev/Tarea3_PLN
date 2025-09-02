import streamlit as st
import requests

# 1. Configuración de la página
st.set_page_config(page_title="El Oráculo del Quijote", page_icon="📜")

st.title("📜 El Oráculo del Quijote 🖋️")
st.write("Pregúntale al modelo fine-tuneado sobre los personajes, eventos y detalles de la obra.")

# 2. Token de Hugging Face
hf_token = None
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
else:
    st.warning("Necesitas un token de Hugging Face para acceder al modelo.")
    hf_token = st.text_input("Ingresa tu token de Hugging Face:", type="password")

# 3. Endpoint del modelo fusionado
API_URL = "https://api-inference.huggingface.co/models/sareizat-dev/qwen-quijote-merged"

def query_model(prompt, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    return response.json()

# 4. Lógica principal
if hf_token:
    pregunta_usuario = st.text_area(
        "Ingresa tu pregunta:",
        "Por ejemplo: ¿Quién es el escudero de Don Quijote?"
    )

    if st.button("Obtener Respuesta"):
        with st.spinner("Consultando al Oráculo..."):
            prompt = f"""
Eres un erudito experto en la obra "Don Quijote de la Mancha" de Miguel de Cervantes.
Tu misión es responder preguntas sobre los personajes, eventos y temas del libro,
utilizando el tono y estilo de la obra. Sé conciso y preciso.

Pregunta: {pregunta_usuario}
Respuesta:
"""
            output = query_model(prompt, hf_token)

            if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                st.success("Respuesta del Oráculo:")
                st.write(output[0]["generated_text"])
            else:
                st.error(f"No se pudo obtener respuesta. Detalles: {output}")

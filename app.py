import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfApi

# 1. Configuración de la página
st.set_page_config(page_title="El Oráculo del Quijote", page_icon="📜")

# 2. Título de la aplicación
st.title("📜 El Oráculo del Quijote 🖋️")
st.write("Pregúntale al modelo fine-tuneado sobre los personajes, eventos y detalles de la obra.")

# 3. Función para cargar el modelo
@st.cache_resource
def cargar_modelo(hf_token):
    try:
        model_name = "sareizat-dev/qwen-quijote-finetuned"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True  # 👈 Añade esta línea
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True) # 👈 Añade esta línea
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error al cargar el modelo. Verifica tu token o la conexión. Detalles: {e}")
        return None, None

# 4. Input para el token de Hugging Face
hf_token = None

if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
else:
    st.warning("Para usar la aplicación, necesitas un token de Hugging Face. El modelo es privado.")
    hf_token = st.text_input("Ingresa tu token de Hugging Face aquí:", type="password")

# 5. Lógica principal de la aplicación
if hf_token:
    api = HfApi()
    try:
        api.whoami(token=hf_token)
        st.success("Token de Hugging Face válido.")
        model, tokenizer = cargar_modelo(hf_token)

        if model is not None and tokenizer is not None:
            pregunta_usuario = st.text_area(
                "Ingresa tu pregunta:",
                "Por ejemplo: ¿Quién es el escudero de Don Quijote?"
            )

            if st.button("Obtener Respuesta"):
                if pregunta_usuario:
                    prompt_chatml = f"""
<|im_start|>system
Eres un erudito experto en la obra "Don Quijote de la Mancha" de Miguel de Cervantes. Tu misión es responder preguntas sobre los personajes, eventos y temas del libro, utilizando el tono y estilo de la obra. Proporciona respuestas concisas y precisas basadas en la obra.
<|im_end|>
<|im_start|>user
{pregunta_usuario}
<|im_end|>
<|im_start|>assistant
"""
                    
                    with st.spinner("Pensando en la respuesta..."):
                        inputs = tokenizer(prompt_chatml, return_tensors="pt").to(model.device)
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=150,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.95,
                                pad_token_id=tokenizer.eos_token_id
                            )
                    
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                    start_token = "<|im_start|>assistant\n"
                    start_index = full_response.rfind(start_token)
                    if start_index != -1:
                        respuesta_asistente = full_response[start_index + len(start_token):].strip()
                        end_token = "<|im_end|>"
                        if end_token in respuesta_asistente:
                            respuesta_asistente = respuesta_asistente.split(end_token)[0].strip()
                        
                        st.success("Respuesta del Oráculo:")
                        st.write(respuesta_asistente)
                    else:
                        st.warning("No pude generar una respuesta clara. Intenta reformular tu pregunta.")
                else:
                    st.warning("Por favor, ingresa una pregunta para continuar.")
    except Exception as e:
        st.error(f"El token no es válido o ha ocurrido un error de conexión: {e}")

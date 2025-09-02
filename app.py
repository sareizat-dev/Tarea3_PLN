import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Configuración de la página
st.set_page_config(page_title="El Oráculo del Quijote", page_icon="📜")

# 2. Título de la aplicación
st.title("📜 El Oráculo del Quijote 🖋️")
st.write("Pregúntale al modelo fine-tuneado sobre los personajes, eventos y detalles de la obra.")

# 3. Cargar el modelo con caché de recursos
@st.cache_resource
def cargar_modelo():
    """Carga el modelo y el tokenizador una sola vez."""
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
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Opcional: ajustar el pad_token de Qwen
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

model, tokenizer = cargar_modelo()

if model is not None and tokenizer is not None:
    # 4. Input del usuario
    pregunta_usuario = st.text_area("Ingresa tu pregunta:", "Por ejemplo: ¿Cuál es la verdadera identidad de Don Quijote?")

    # 5. Lógica de la respuesta del modelo
    if st.button("Obtener Respuesta"):
        if pregunta_usuario:
            # Crear el prompt usando el formato ChatML
            prompt_chatml = f"""
<|im_start|>system
Eres un erudito experto en la obra "Don Quijote de la Mancha" de Miguel de Cervantes. Tu misión es responder preguntas sobre los personajes, eventos y temas del libro, utilizando el tono y estilo de la obra. Proporciona respuestas concisas y precisas basadas en la obra.
<|im_end|>
<|im_start|>user
{pregunta_usuario}
<|im_end|>
<|im_start|>assistant
"""
            
            # Codificar el prompt y generar la respuesta
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
            
            # Decodificar y formatear la respuesta
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

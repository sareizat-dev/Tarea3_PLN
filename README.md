# 📜 El Oráculo del Quijote 🖋️

## Descripción del Proyecto

**El Oráculo del Quijote** es una aplicación web interactiva que utiliza un modelo de lenguaje de inteligencia artificial fine-tuneado para responder preguntas sobre la aclamada obra de Miguel de Cervantes, **"Don Quijote de la Mancha"**.

El corazón de este proyecto es el modelo **Qwen1.5-1.8B-Chat** de Qwen, que ha sido **sintonizado (fine-tuned)** con el texto completo del libro. Gracias a este entrenamiento especializado, el modelo ha aprendido el estilo, los personajes y los eventos de la obra, permitiéndole generar respuestas precisas y contextualmente relevantes a las preguntas del usuario.

## Características

* **Respuestas Contextuales**: El modelo responde a las preguntas utilizando el conocimiento adquirido de la obra, ofreciendo información sobre personajes, tramas, y la filosofía del libro.
* **Interacción Sencilla**: La interfaz de usuario, construida con **Streamlit**, es intuitiva y fácil de usar, permitiendo a cualquier persona interactuar con el modelo sin necesidad de conocimientos técnicos.
* **Despliegue Eficiente**: El proyecto utiliza **Git LFS** para gestionar el modelo de gran tamaño en GitHub y se despliega en **Streamlit Cloud**, optimizando la carga y el rendimiento de la aplicación.
* **QLoRA Fine-Tuning**: El modelo fue entrenado con **QLoRA** (Quantized Low-Rank Adaptation), una técnica de fine-tuning eficiente que permite adaptar modelos grandes con recursos de hardware limitados.

## Estructura del Repositorio

.
├── .streamlit/  # Carpeta opcional de configuración de Streamlit
├── app.py       # Archivo principal de la aplicación
├── requirements.txt # Dependencias de Python
├── qwen-quijote-finetuned/ # Carpeta del modelo fine-tuneado (rastreado por Git LFS)
└── README.md

## Guía de Uso

Puedes acceder a la aplicación en vivo [aquí](https://tu-usuario.streamlit.app) (¡reemplaza este enlace con el de tu aplicación desplegada!).

Para usar la aplicación:
1.  Ingresa tu pregunta sobre la obra en el cuadro de texto.
2.  Haz clic en el botón "Obtener Respuesta".
3.  Espera unos segundos mientras el modelo genera la respuesta.

## Cómo Funciona

1.  **Modelo Fine-Tuneado**: El modelo base Qwen1.5-1.8B-Chat fue sintonizado con el libro "Don Quijote de la Mancha" en un entorno de **Google Colab**, utilizando una GPU T4 y la técnica QLoRA para optimizar el uso de recursos.
2.  **Manejo de Archivos Grandes**: La carpeta del modelo (`qwen-quijote-finetuned/`) se almacena en este repositorio utilizando **Git LFS**, lo que permite subir archivos que exceden el límite de tamaño de GitHub.
3.  **Despliegue con Streamlit Cloud**: Streamlit Cloud clona este repositorio, lee el archivo `requirements.txt` para instalar las dependencias y ejecuta `app.py` para poner la aplicación en línea. La función de caché de Streamlit (`@st.cache_resource`) asegura que el modelo solo se cargue una vez por sesión, mejorando la experiencia del usuario.

## Requisitos y Tecnologías

* **Lenguaje de Programación**: Python
* **Modelo Base**: [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
* **Framework de IA**: `transformers`, `peft`, `bitsandbytes`, `trl`
* **Interfaz de Usuario**: Streamlit
* **Control de Versiones**: Git LFS

## Contacto y Contribución

Si deseas contribuir al proyecto o tienes alguna pregunta, no dudes en abrir un *issue* o una *pull request* en este repositorio.

---
© 2025 [sareizat-dev]. Todos los derechos reservados.

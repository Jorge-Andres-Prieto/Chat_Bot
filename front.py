# Importa Streamlit para construir interfaces de usuario
import streamlit as st

# Importa funciones del módulo chat_bot
from chat_bot import predict_class, get_response, intents

# Configura el título de la aplicación de Streamlit
st.title('🤖 UNalBot')

# Inicializa el estado de la sesión para almacenar mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Configura la barra lateral con preguntas frecuentes
st.sidebar.title("Preguntas Frecuentes")
st.sidebar.write("1. ¿Cómo puedo contactar a soporte?")
st.sidebar.write("R: Puedes contactarnos vía email en soporte@unal.edu.co")
st.sidebar.write("2. ¿Dónde encuentro información sobre matrículas?")
st.sidebar.write("R: Toda la información sobre matrículas está disponible en nuestro portal web.")

# Muestra los mensajes guardados en el estado de la sesión
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Envía el primer mensaje del asistente si es la primera interacción
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({
        "role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"
    })
    st.session_state.first_message = False

# Captura y maneja la entrada del usuario
prompt = st.chat_input("¿cómo puedo ayudarte?")
if prompt:
    # Agrega y muestra el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Procesa la entrada del usuario usando el modelo de IA
    intents_list = predict_class(prompt)
    response = get_response(intents_list, intents)

    # Envía y muestra la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

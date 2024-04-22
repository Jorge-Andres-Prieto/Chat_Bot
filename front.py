import streamlit as st
from chat_bot import predict_class, get_response, intents

# Configura el título de la aplicación de Streamlit
st.title('🤖 UNalBot')

# Inicializa el estado de la sesión para almacenar mensajes si aún no está hecho
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

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

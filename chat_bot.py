import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from tensorflow.keras.optimizers import SGD  # Solo si decido recompilar el modelo


def setup_nltk():
    """Configura los recursos necesarios de NLTK.

    Descarga los recursos 'punkt', 'wordnet' y 'omw-1.4' si no están
    presentes en el entorno de ejecución. Esto es necesario para asegurar
    que las funciones de NLTK que dependen de estos recursos puedan
    operar correctamente.
    """
    nltk_resources = ['punkt', 'wordnet', 'omw-1.4']
    for resource in nltk_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


# Llamar a setup_nltk al inicio para asegurar la disponibilidad de recursos
setup_nltk()

# Inicialización del lematizador y carga de datos
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Carga del modelo
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Procesa la oración para tokenización y lematización.

    Args:
        sentence (str): La oración a procesar.

    Returns:
        list: Lista de palabras lematizadas.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convierte una oración en una bolsa de palabras.

    Args:
        sentence (str): Oración a convertir.

    Returns:
        numpy.array: Representación binaria de la bolsa de palabras.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        if s_word in words:
            bag[words.index(s_word)] = 1
    return np.array([bag])

def predict_class(sentence):
    """Predice la clase de intención para una oración dada.

    Args:
        sentence (str): Oración a clasificar.

    Returns:
        list: Lista de diccionarios con la intención y la probabilidad.
    """
    bow = bag_of_words(sentence)
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Genera una respuesta basada en la lista de intenciones clasificada.

    Args:
        intents_list (list): Lista de intenciones predichas.
        intents_json (dict): JSON con todas las intenciones y respuestas.

    Returns:
        str: Una respuesta seleccionada al azar para la intención.
    """
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


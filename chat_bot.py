# Importa random para seleccionar elementos al azar
import random

# Importa json para manipulación de archivos JSON
import json
# Importa pickle para serialización de objetos
import pickle
# Importa NumPy para manipulación de arrays
import numpy as np

# Importa NLTK para procesamiento de texto
import nltk
# Importa WordNetLemmatizer para lematización
from nltk.stem import WordNetLemmatizer

# Importa load_model para cargar modelos de Keras
from keras.models import load_model
# Importa SGD para configuración de optimizadores
from tensorflow.keras.optimizers import SGD

def setup_nltk():
    """
    Configura los recursos necesarios de NLTK.

    Descarga los recursos 'punkt', 'wordnet' y 'omw-1.4' si no están
    presentes en el entorno de ejecución para asegurar la funcionalidad
    de procesamiento de texto de NLTK.

    Args:
        None

    Returns:
        None
    """
    nltk_resources = ['punkt', 'wordnet', 'omw-1.4']
    for resource in nltk_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

# Inicializa NLTK para asegurar la disponibilidad de los recursos
setup_nltk()

# Inicializa el lematizador y carga el archivo de intenciones
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Carga las palabras y clases desde archivos previamente guardados
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Carga el modelo previamente entrenado
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """
    Procesa una oración para tokenización y lematización.

    Args:
        sentence (str): La oración a procesar.

    Returns:
        list: Lista de palabras lematizadas de la oración.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    Convierte una oración en una bolsa de palabras.

    Args:
        sentence (str): La oración a convertir.

    Returns:
        numpy.array: Array que representa la bolsa de palabras, binaria.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        if s_word in words:
            bag[words.index(s_word)] = 1
    return np.array([bag])

def predict_class(sentence):
    """
    Predice la clase de intención de una oración dada.

    Args:
        sentence (str): Oración a clasificar.

    Returns:
        list: Lista de diccionarios con la intención y probabilidad asociada.
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
    """
    Genera una respuesta basada en la lista de intenciones clasificada.

    Args:
        intents_list (list): Lista de intenciones clasificadas.
        intents_json (dict): JSON que contiene todas las intenciones y respuestas.

    Returns:
        str: Respuesta seleccionada al azar basada en la intención detectada.
    """
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Importa el módulo random para generar números aleatorios
import random

# Importa el módulo json para manipular datos en formato JSON
import json

# Importa el módulo pickle para serializar y deserializar objetos
import pickle

# Importa NumPy para operaciones numéricas avanzadas
import numpy as np

# Importa NLTK para procesamiento de lenguaje natural
import nltk
# Importa WordNetLemmatizer para lematización
from nltk.stem import WordNetLemmatizer

# Importa Sequential para construir modelos de red neuronal
from keras.models import Sequential
# Importa capas para la red neuronal
from keras.layers import Dense, Activation, Dropout

# Importa SGD para optimización del modelo
from tensorflow.keras.optimizers import SGD


def setup_nltk():
    """
    Configura los recursos necesarios de NLTK.

    Esta función descarga los recursos 'punkt', 'wordnet' y 'omw-1.4' si no están
    presentes en el entorno de ejecución. Es necesario para asegurar
    que las funciones de NLTK operen correctamente.

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


# Llamada inicial para configurar NLTK
setup_nltk()

# Cargar y preparar datos para el modelo
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Descargar y asegurar disponibilidad de recursos NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preparar estructuras de datos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']  # Símbolos a ignorar en el procesamiento

# Procesamiento de cada intención y patrón
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada patrón
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematización y limpieza de palabras
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar datos procesados para uso futuro
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparación de datos para entrenamiento del modelo
training = []
output_empty = [0] * len(classes)  # Lista inicial para la salida del modelo
for document in documents:
    # Crear una bolsa de palabras para cada documento
    bag = [0] * len(words)
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar los datos de entrenamiento y convertir a arrays de NumPy
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Definir y configurar el modelo de red neuronal
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(words),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Configurar el optimizador
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guardar el modelo entrenado para uso posterior
model.save('chatbot_model.h5')

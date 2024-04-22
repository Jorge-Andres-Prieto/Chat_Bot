import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


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

# Cargar y preparar datos
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preparar estructuras de datos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Procesar cada intent y patrón
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenización de patrones
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematización y limpieza de palabras
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar datos procesados
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparación de datos para entrenamiento
training = []
output_empty = [0] * len(classes)
for document in documents:
    # Inicialización de la bolsa de palabras
    bag = [0] * len(words)
    # Lematización y normalización
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1
    # Preparación de la salida del entrenamiento
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar aleatoriamente los datos de entrenamiento
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Definir y compilar el modelo
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(words),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Configuración del optimizador
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamiento del modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5')

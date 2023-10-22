import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Texto de entrada
texto = "Este es un ejemplo de texto que contiene algunos caracteres especiales, números y palabras en mayúsculas. Queremos preprocesar este texto para que esté listo para el análisis de texto en un proyecto de Procesamiento de Lenguaje Natural (NLP)."

# Eliminación de caracteres especiales y números
texto = re.sub(r'[^a-zA-Z]', ' ', texto)

# Conversión a minúsculas
texto = texto.lower()

# Tokenización
palabras = nltk.word_tokenize(texto)

# Eliminación de stopwords
stop_words = set(stopwords.words('spanish'))  # Puedes ajustar el idioma
palabras = [palabra for palabra in palabras if palabra not in stop_words]

# Stemming
stemmer = PorterStemmer()
palabras = [stemmer.stem(palabra) for palabra in palabras]

# Resultado
texto_procesado = ' '.join(palabras)
print(texto_procesado)

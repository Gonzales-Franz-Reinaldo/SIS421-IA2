import os
import re
import nltk
from nltk.corpus import stopwords

# Descargar las stopwords de NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))  # Cambia 'spanish' si usas otro idioma

# Ruta de la carpeta con los archivos originales
input_folder = r'C:\Dataset_textos\dataset_chatbot'
# Ruta de la carpeta donde se guardarán los archivos procesados
output_folder = r'C:\Dataset_textos\textos_procesados'
os.makedirs(output_folder, exist_ok=True)

def limpiar_y_tokenizar(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales, números y signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)  # Elimina signos de puntuación
    text = re.sub(r'\d+', '', text)  # Elimina números
    # Tokenizar y eliminar stopwords
    tokens = text.split()  # Divide el texto en palabras
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)  # Devuelve el texto limpio como una cadena

# Procesar cada archivo .txt
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)
        try:
            # Intentar leer el archivo con codificación UTF-8
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Si falla, intenta leer el archivo con codificación ISO-8859-1 (Latin-1)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                text = file.read()
        
        # Limpiar y tokenizar
        cleaned_text = limpiar_y_tokenizar(text)
        # Guardar el texto limpio con el mismo nombre en la carpeta de salida
        with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_text)

import os
import re
from collections import Counter

# Ruta de la carpeta con los archivos procesados
input_folder = r'C:\Dataset_textos\textos_procesados'
# Ruta del archivo donde se guardará el vocabulario
vocabulario_file = r'C:\Dataset_textos\extraer_vocabulario\vocabulario.txt'
os.makedirs(os.path.dirname(vocabulario_file), exist_ok=True)

def extraer_vocabulario(text):
    # Dividir el texto en palabras (tokens)
    tokens = text.split()
    # Contar la ocurrencia de cada palabra
    return Counter(tokens)

# Diccionario para almacenar todas las palabras y sus conteos
vocabulario_total = Counter()

# Procesar cada archivo .txt en la carpeta procesada
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Extraer y agregar al vocabulario total
            vocabulario_total.update(extraer_vocabulario(text))

# Guardar el vocabulario único en un archivo de texto
with open(vocabulario_file, 'w', encoding='utf-8') as vocab_file:
    for word, count in vocabulario_total.most_common():
        vocab_file.write(f"{word}\n")

import os

# Ruta del archivo del vocabulario
vocabulario_file = r'C:\Dataset_textos\extraer_vocabulario\vocabulario.txt'
# Ruta de la carpeta con los archivos procesados
input_folder = r'C:\Dataset_textos\textos_procesados'
# Ruta de la carpeta donde se guardarán las secuencias numéricas
output_folder = r'C:\Dataset_textos\secuencias_numericas'
os.makedirs(output_folder, exist_ok=True)

# Cargar el vocabulario y asignar un índice a cada palabra
vocabulario = {}
with open(vocabulario_file, 'r', encoding='utf-8') as file:
    for idx, word in enumerate(file):
        vocabulario[word.strip()] = idx + 1  # Asigna el índice desde 1 (0 se suele reservar para padding)

# Función para convertir el texto en secuencias numéricas
def convertir_a_secuencia(text):
    # Dividir el texto en palabras y convertir cada una a su índice correspondiente
    return [vocabulario[word] for word in text.split() if word in vocabulario]

# Procesar cada archivo de texto y guardar la secuencia numérica
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Convertir el texto en una secuencia numérica
            secuencia_numerica = convertir_a_secuencia(text)
            # Guardar la secuencia en un archivo nuevo
            with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as output_file:
                output_file.write(" ".join(map(str, secuencia_numerica)))

import os
import numpy as np
import pandas as pd

# Ruta de la carpeta con las secuencias numéricas
input_folder = r'C:\Dataset_textos\secuencias_numericas'
# Ruta para guardar el archivo estructurado del dataset
output_file = r'C:\Dataset_textos\dataset_textos.csv'

# Definir la longitud máxima de las secuencias
max_length = 20  # Puedes ajustar esto según tu preferencia y necesidades

# Función para aplicar padding o truncar secuencias
def ajustar_longitud(secuencia, max_length):
    if len(secuencia) > max_length:
        return secuencia[:max_length]
    else:
        return secuencia + [0] * (max_length - len(secuencia))  # Padding con 0s

# Almacenar pares de entrada-respuesta
pairs = []

# Leer los archivos y agrupar en pares
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.txt'):
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
            # Lee cada archivo como una lista de secuencias numéricas
            secuencias = [list(map(int, line.split())) for line in file]
            
            # Crear pares entrada-respuesta
            for i in range(len(secuencias) - 1):
                entrada = ajustar_longitud(secuencias[i], max_length)
                respuesta = ajustar_longitud(secuencias[i + 1], max_length)
                pairs.append((entrada, respuesta))

# Convertir a DataFrame y guardar
df = pd.DataFrame(pairs, columns=["entrada", "respuesta"])
df.to_csv(output_file, index=False)

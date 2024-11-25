import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import pyttsx3
import pywhatkit  # Para reproducir canciones en YouTube

# Crear un archivo temporal para guardar el audio capturado
temp_file = tempfile.mkdtemp()
save_path = os.path.join(temp_file, 'audio.wav')

# Inicializar el reconocimiento de voz
listener = sr.Recognizer()

# Configurar el motor de texto a voz (pyttsx3)
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Descomentar para inspeccionar las voces disponibles
# print(f"Voces disponibles: {len(voices)}")
# for index, voice in enumerate(voices):
#     print(f"Voz {index}: {voice.name} - {voice.languages}")

# Configurar la velocidad del habla y seleccionar la primera voz disponible
engine.setProperty('rate', 140)  # Velocidad del habla (ajustable)
engine.setProperty('voice', voices[0].id)  # Seleccionar una voz

# Función para que el sistema hable el texto recibido
def talk(text):
    engine.say(text)
    engine.runAndWait()

# Función para escuchar audio del micrófono y guardarlo como un archivo WAV
def listen():
    try:
        with sr.Microphone() as source:
            print('Por favor di algo: ')
            
            # Ajustar el micrófono para reducir ruido ambiental
            listener.adjust_for_ambient_noise(source)
            
            # Capturar el audio del usuario
            audio = listener.listen(source)
            
            # Convertir el audio capturado a formato WAV
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data).remove_dc_offset().normalize()
            
            # Guardar el archivo de audio procesado en el directorio temporal
            audio_clip.export(save_path, format='wav', bitrate="192k")
    except Exception as e:
        print(e) 
    return save_path



# Función para transcribir el audio usando Whisper
def recognize_audio(audio_path):
    # Cargar el modelo Whisper (puede ser 'base', 'small', 'medium', o 'large')
    model = whisper.load_model('medium')  
    
    # Opciones de transcripción
    options = {"language": "es", "fp16": False, "temperature": 0.0}
    
    # Transcribir el audio y devolver el texto
    transcription = model.transcribe(audio_path, **options)
    return transcription['text']

# Función principal para ejecutar el flujo del programa
def main():
    try:
        # Capturar y procesar el audio del usuario
        audio_path = listen()
        
        # Obtener la transcripción del audio
        text_response = recognize_audio(audio_path)
        
        # Comprobar si el comando es para reproducir música
        if 'reproduce' in text_response or 'Reproduce' in text_response:
            song = text_response.replace('reproduce', '')  # Extraer el nombre de la canción
            talk(f'Reproduciendo {song}')  # Informar al usuario
            pywhatkit.playonyt(song)  # Reproducir la canción en YouTube
            print(text_response)  
            return
        else:
            # Si no se reconoce un comando, simplemente hablar el texto transcrito
            talk(text_response)
            print(text_response)  
            return
    except Exception as e:
        print(e)  
        talk('Lo siento, no te he entendido')
        return

# Punto de entrada del programa
if __name__ == '__main__':
    main()

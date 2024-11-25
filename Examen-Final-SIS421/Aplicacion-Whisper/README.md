# Comando a ejecutar en system terminal
- python -m venv venv
- .\venv\Scripts\activate
- pip install  pyaudio
- pip install  pydub
- pip install SpeechRecognition

# Instalamos libreria para audios
- pip install pyttsx3
- pip install pydub

# Para preubas extra
- pip install pywhatkit


(venv) C:\GONZALES\Gonzales-CICO\SIS421-IA2\Examen-Final\Aplicacion-Whisper>pip freeze
certifi==2024.8.30
charset-normalizer==3.4.0
idna==3.10
PyAudio==0.2.14
requests==2.32.3
SpeechRecognition==3.11.0
typing_extensions==4.12.2
urllib3==2.2.3


# En powerSell
- Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))


# Windows- en terminal
- choco install ffmpeg


# Instalamos Whisper
- pip install git+https://github.com/openai/whisper.git 

# Instalamos libreria para audios
- pip install pyttsx3
- pip install pydub

# Para preubas extra
- pip install pywhatkit
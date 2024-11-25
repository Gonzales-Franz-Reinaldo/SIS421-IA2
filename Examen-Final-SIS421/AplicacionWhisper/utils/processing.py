def process_audio(task, filepath, model):
    """
    Procesa un archivo de audio con Whisper según la tarea especificada.

    Args:
        task (str): Tarea a realizar ('asr', 'detect_language', 'translate', 'multilingual_transcription').
        filepath (str): Ruta del archivo de audio.
        model: Modelo preentrenado Whisper.

    Returns:
        dict: Resultado de la tarea procesada o un error.
    """
    try:
        if task == 'asr':  # Reconocimiento Automático del Habla
            result = model.transcribe(filepath)
            
        elif task == 'detect_language':  # Detección del Idioma
            result = model.transcribe(filepath)
            language = result.get('language', 'Idioma no detectado')
            return {'text': f"Idioma detectado: {language}"}
        
        elif task == 'translate':  # Traducción a Inglés
            result = model.transcribe(filepath, task="translate")
            
        elif task == 'multilingual_transcription':  # Transcripción Multilingüe
            result = model.transcribe(filepath)
            
        return {'text': result['text']}  # Devuelve el texto procesado
    except Exception as e:  # Manejo de excepciones
        return {'error': str(e)}  # Devuelve el error como texto

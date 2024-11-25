import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
import whisper  # Librería para el modelo Whisper
from werkzeug.utils import secure_filename
from utils.processing import process_audio  # Función para procesar audio
from utils.export import export_to_txt, export_to_pdf  # Funciones para exportar resultados


# Configuración de la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Carpeta para guardar archivos
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Tamaño máximo permitido de 50 MB
app.secret_key = 'supersecretkey'  # Clave para manejar sesiones y mensajes


# Carga el modelo Whisper (modelo preentrenado "small")
model = whisper.load_model("small")


# Ruta principal para tareas específicas
@app.route('/task/<string:task>', methods=['GET', 'POST'])
def task_page(task):
    if request.method == 'POST':  # Si se envía un formulario
        # Verifica si se subió un archivo cargado o grabado
        if 'audio' in request.files:  # Caso de archivo cargado
            file = request.files['audio']
            if file.filename == '':  # Validación de archivo vacío
                flash("El archivo está vacío.")
                return redirect(request.url)

            # Guardar el archivo cargado
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Archivo cargado guardado en: {filepath}")

        elif 'recorded_audio' in request.files:  # Caso de grabación
            file = request.files['recorded_audio']

            # Guardar archivo grabado como temporal
            filename = "recorded_audio.wav"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Archivo grabado guardado en: {filepath}")

        else:  # Si no se subió ningún archivo
            flash("No se seleccionó ningún archivo.")
            return redirect(request.url)

        # Genera URL para previsualización
        audio_url = url_for('static', filename=f'uploads/{filename}')

        #! Procesa el audio con el modelo Whisper según la tarea
        result = process_audio(task, filepath, model)
        
        if 'error' in result:  # Manejo de errores durante el procesamiento
            flash(result['error'])
            return redirect(request.url)

        # Exporta resultados en formato TXT y PDF
        text = result['text']
        export_to_txt(text, app.config['UPLOAD_FOLDER'], "result.txt")
        export_to_pdf(text, app.config['UPLOAD_FOLDER'], "result.pdf")

        # Renderiza la plantilla con los resultados y previsualización del audio
        return render_template(
            f'{task}.html',
            task=task,
            result=text,
            audio_url=audio_url  # URL para reproducir el audio
        )

    # Renderiza la plantilla de la tarea al acceder por GET
    return render_template(f'{task}.html')

# Ruta para servir archivos desde static/uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ruta para descargar los resultados exportados
@app.route('/download/<string:filetype>')
def download_file(filetype):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"result.{filetype}")
    if os.path.exists(filepath):  # Verifica si el archivo existe
        return send_file(filepath, as_attachment=True)  # Descarga como adjunto
    flash("El archivo solicitado no está disponible.")  # Error si no existe
    return redirect(url_for('index'))

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Crea la carpeta de uploads si no existe
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)  # Ejecuta la aplicación en modo debug

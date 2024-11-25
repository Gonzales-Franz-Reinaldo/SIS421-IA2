let mediaRecorder;
let audioChunks = [];
let timerInterval;
let seconds = 0;

// Inicializa los elementos del DOM
function initializeAudioRecorder(recordBtnId, timerDisplayId, formId) {
    const recordBtn = document.getElementById(recordBtnId);
    const timerDisplay = document.getElementById(timerDisplayId);
    const form = document.getElementById(formId);

    recordBtn.addEventListener("click", () => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            startRecording(recordBtn, timerDisplay);
        } else {
            stopRecording(recordBtn, timerDisplay, form);
        }
    });
}

function startRecording(recordBtn, timerDisplay) {
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const file = new File([audioBlob], "recorded_audio.wav", {
                type: "audio/wav",
            });

            // Crear un nuevo formulario para enviar el archivo grabado
            const formData = new FormData();
            formData.append("audio", file);

            // Enviar el archivo al backend
            fetch(window.location.href, {
                method: "POST",
                body: formData,
            })
                .then((response) => response.text())
                .then((html) => {
                    document.body.innerHTML = html; // Carga la nueva respuesta en la página
                })
                .catch((error) =>
                    console.error("Error al enviar el archivo grabado:", error)
                );
        };

        startTimer(timerDisplay);
        recordBtn.innerHTML =
            '<i class="material-icons text-2xl text-red-500">stop</i>';
    });
}

function stopRecording(recordBtn, timerDisplay, form) {
    mediaRecorder.stop();
    stopTimer();
    recordBtn.innerHTML = '<i class="material-icons text-2xl">mic</i>';
}

function startTimer(timerDisplay) {
    seconds = 0;
    timerDisplay.textContent = "00:00";
    timerInterval = setInterval(() => {
        seconds++;
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        timerDisplay.textContent = `${String(minutes).padStart(2, "0")}:${String(
            secs
        ).padStart(2, "0")}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}

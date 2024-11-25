import os
from functools import lru_cache  # Decorador para implementar una caché de almacenamiento
from subprocess import CalledProcessError, run  # Ejecuta comandos externos (usado para llamar a ffmpeg)
from typing import Optional, Union

import numpy as np  # Biblioteca para manipulación de datos numéricos
import torch  # Biblioteca para computación en GPU
import torch.nn.functional as F  # Funciones de utilidad para redes neuronales

from .utils import exact_div  # Función personalizada para divisiones exactas

# **Parámetros fijos del audio**
SAMPLE_RATE = 16000  # Frecuencia de muestreo del audio en Hz
N_FFT = 400  # Número de puntos FFT (Fast Fourier Transform) para calcular STFT
HOP_LENGTH = 160  # Espaciado entre ventanas en STFT
CHUNK_LENGTH = 30  # Longitud máxima de cada segmento de audio en segundos
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # Total de muestras en un segmento de 30 segundos
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # Número de frames en el espectrograma Mel

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # Muestras por token generado
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # Frames de audio por segundo
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # Tokens generados por segundo



def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Abre un archivo de audio y lo convierte en un waveform mono, remuestreándolo si es necesario.
    """
    cmd = [
        "ffmpeg",  # Utiliza FFmpeg para procesar el archivo de audio
        "-nostdin",  # Desactiva el stdin
        "-threads", "0",  # Usa todos los hilos disponibles
        "-i", file,  # Archivo de entrada
        "-f", "s16le",  # Formato de salida: audio sin procesar (PCM)
        "-ac", "1",  # Convierte el audio a mono
        "-acodec", "pcm_s16le",  # Codificación en PCM de 16 bits
        "-ar", str(sr),  # Cambia la frecuencia de muestreo
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout  # Ejecuta el comando y captura la salida
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0  # Convierte a float32





def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Ajusta el tamaño del arreglo de audio a `N_SAMPLES` añadiendo ceros o recortando.
    """
    if torch.is_tensor(array):  # Si el arreglo es un tensor de PyTorch
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:  # Si es un arreglo NumPy
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array






@lru_cache(maxsize=None)  # Almacena en caché para evitar recomputación
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    Carga la matriz de filtros Mel para proyectar el STFT en un espectrograma Mel.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)





def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Calcula el espectrograma Mel logarítmico a partir de una señal de audio.
    """
    if not torch.is_tensor(audio):  # Si el audio no es un tensor, conviértelo
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    # Aplica STFT (Transformada de Fourier de Tiempo Corto)
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2  # Magnitud del espectro

    # Proyección a escala Mel
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    # Conversión a escala logarítmica
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


"""Entrada / salida de audio: cargar, guardar, convertir formato y reproducir."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

#: Formatos admitidos al guardar o convertir (extensión -> formato de libsndfile).
FORMATOS = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG", ".mp3": "MP3"}

#: Frecuencias de muestreo que admite el formato MP3.
_FS_MP3 = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


def remuestrear(x: np.ndarray, fs_orig: int, fs_dest: int) -> np.ndarray:
    """Cambia la frecuencia de muestreo con un filtro polifásico antialiasing."""
    if int(fs_orig) == int(fs_dest):
        return np.asarray(x, dtype=float).copy()
    r = Fraction(int(fs_dest), int(fs_orig)).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=float), r.numerator, r.denominator)


def normalizar(x: np.ndarray, pico: float = 0.95) -> np.ndarray:
    """Escala la señal para que su valor absoluto máximo sea ``pico``."""
    x = np.asarray(x, dtype=float)
    maximo = np.max(np.abs(x)) if x.size else 0.0
    return x.copy() if maximo == 0 else x * (pico / maximo)


def cargar_audio(ruta, fs_objetivo: int | None = None, mono: bool = True):
    """Lee un archivo de audio y devuelve ``(x, fs)`` con muestras en [-1, 1].

    Si el archivo es estéreo y ``mono`` es verdadero se promedian los canales.
    Si se indica ``fs_objetivo`` la señal se remuestrea a esa frecuencia.
    """
    x, fs = sf.read(str(ruta), dtype="float64", always_2d=True)
    x = x.mean(axis=1) if mono else x
    if fs_objetivo and fs_objetivo != fs:
        x = remuestrear(x, fs, fs_objetivo)
        fs = fs_objetivo
    return x, int(fs)


def guardar_audio(ruta, x: np.ndarray, fs: int) -> Path:
    """Guarda la señal en el formato indicado por la extensión del archivo.

    Admite WAV (PCM 16 bits), FLAC, OGG Vorbis y MP3.
    """
    ruta = Path(ruta)
    extension = ruta.suffix.lower()
    if extension not in FORMATOS:
        raise ValueError(
            f"Formato no admitido: {extension or '(sin extensión)'}. "
            f"Usa uno de: {', '.join(FORMATOS)}"
        )
    datos = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)

    if extension == ".wav":
        sf.write(ruta, datos, fs, format="WAV", subtype="PCM_16")
    elif extension == ".flac":
        sf.write(ruta, datos, fs, format="FLAC", subtype="PCM_16")
    elif extension == ".ogg":
        sf.write(ruta, datos, fs, format="OGG", subtype="VORBIS")
    else:  # .mp3
        if fs not in _FS_MP3:
            datos = np.clip(remuestrear(datos, fs, 44100), -1.0, 1.0)
            fs = 44100
        sf.write(ruta, datos, fs, format="MP3", subtype="MPEG_LAYER_III")
    return ruta


def convertir_archivo(origen, destino, fs: int | None = None) -> Path:
    """Convierte un archivo de audio a otro formato (y opcionalmente otra fs)."""
    x, fs_leida = cargar_audio(origen, fs_objetivo=fs)
    return guardar_audio(destino, x, fs_leida)


def reproducir(x: np.ndarray, fs: int) -> None:
    """Reproduce la señal por la salida de audio predeterminada (sin bloquear)."""
    import sounddevice as sd

    sd.stop()
    sd.play(np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), int(fs))


def detener() -> None:
    """Detiene la reproducción en curso."""
    import sounddevice as sd

    sd.stop()

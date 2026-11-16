"""Análisis a corto plazo de la señal de voz.

La voz es una señal no estacionaria, pero puede considerarse
cuasi-estacionaria en intervalos de 20-40 ms. Por eso se divide en tramas
(enventanado) y se calculan parámetros sobre cada una:

- Preénfasis: filtro paso alto que compensa la caída espectral de -6 dB/octava
  debida al pulso glotal y la radiación labial.
- Energía a corto plazo y tasa de cruces por cero (ZCR).
- Clasificación de cada trama en silencio, sorda o sonora.
- Espectrograma (STFT) de banda ancha o estrecha.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

SILENCIO, SORDO, SONORO = 0, 1, 2
NOMBRES_CLASE = {SILENCIO: "silencio", SORDO: "sorda", SONORO: "sonora"}

_VENTANAS = {
    "rectangular": "boxcar",
    "hamming": "hamming",
    "hann": "hann",
    "blackman": "blackman",
}
#: Nombres de ventana admitidos.
VENTANAS = tuple(_VENTANAS)


def muestras(ms: float, fs: int) -> int:
    """Convierte una duración en milisegundos a número de muestras (>= 1)."""
    return max(1, int(round(ms * fs / 1000.0)))


def crear_ventana(nombre: str, n: int) -> np.ndarray:
    """Devuelve una ventana simétrica de ``n`` muestras."""
    try:
        return get_window(_VENTANAS[nombre], n, fftbins=False)
    except KeyError:
        raise ValueError(
            f"Ventana desconocida: {nombre!r}. Opciones: {', '.join(VENTANAS)}"
        ) from None


def preenfasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Aplica el filtro de preénfasis ``y[n] = x[n] - coef * x[n-1]``."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    return np.concatenate(([x[0]], x[1:] - coef * x[:-1]))


def enventanar(x, fs, dur_ms=30.0, salto_ms=10.0, ventana="hamming"):
    """Divide la señal en tramas solapadas y aplica una ventana.

    Devuelve ``(tramas, tiempos)``, donde ``tramas`` tiene forma
    ``(n_tramas, muestras_por_trama)`` y ``tiempos`` es el instante central
    de cada trama en segundos.
    """
    x = np.asarray(x, dtype=float)
    n = muestras(dur_ms, fs)
    h = muestras(salto_ms, fs)
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    n_tramas = 1 + (x.size - n) // h
    inicios = h * np.arange(n_tramas)
    tramas = x[inicios[:, None] + np.arange(n)[None, :]]
    tramas = tramas * crear_ventana(ventana, n)
    tiempos = (inicios + n / 2) / fs
    return tramas, tiempos


def energia(tramas: np.ndarray) -> np.ndarray:
    """Energía de cada trama: suma de las muestras al cuadrado."""
    return np.sum(tramas**2, axis=1)


def energia_db(tramas: np.ndarray) -> np.ndarray:
    """Energía de cada trama en dB."""
    return 10 * np.log10(energia(tramas) + 1e-12)


def tasa_cruces_cero(tramas: np.ndarray) -> np.ndarray:
    """Fracción de pares de muestras consecutivas con cambio de signo."""
    signos = np.signbit(tramas)
    return np.mean(signos[:, 1:] != signos[:, :-1], axis=1)


def clasificar_tramas(
    x,
    fs,
    dur_ms=30.0,
    salto_ms=10.0,
    umbral_silencio_db=-35.0,
    umbral_zcr_hz=3000.0,
):
    """Clasifica cada trama como silencio, sorda o sonora.

    - Silencio: energía más de ``umbral_silencio_db`` por debajo de la trama
      más energética.
    - Sonora: hay energía y pocos cruces por cero (la energía se concentra en
      baja frecuencia, como en las vocales).
    - Sorda: hay energía y muchos cruces por cero (ruido de alta frecuencia,
      como en las fricativas /s/, /f/).

    El umbral de cruces por cero se expresa en cruces por segundo para que no
    dependa de la frecuencia de muestreo.

    Devuelve ``(clases, tiempos, energia_relativa_db, zcr_por_segundo)``.
    """
    tramas, tiempos = enventanar(x, fs, dur_ms, salto_ms, "rectangular")
    e_db = energia_db(tramas * crear_ventana("hamming", tramas.shape[1]))
    e_rel = e_db - e_db.max()
    zcr_hz = tasa_cruces_cero(tramas) * fs

    clases = np.full(len(tiempos), SORDO)
    clases[e_rel < umbral_silencio_db] = SILENCIO
    clases[(e_rel >= umbral_silencio_db) & (zcr_hz < umbral_zcr_hz)] = SONORO
    return clases, tiempos, e_rel, zcr_hz


def segmento_sonoro(x, fs, dur_ms=40.0):
    """Extrae un segmento de ``dur_ms`` centrado en la trama sonora más fuerte.

    Se usa para los análisis que necesitan un trozo estacionario de vocal
    (envolvente LPC, flujo glotal, cepstrum). Devuelve ``(segmento, t_inicio)``.
    """
    x = np.asarray(x, dtype=float)
    clases, tiempos, e_rel, _ = clasificar_tramas(x, fs)
    candidatas = np.flatnonzero(clases == SONORO)
    if candidatas.size == 0:
        candidatas = np.arange(len(tiempos))
    i = candidatas[np.argmax(e_rel[candidatas])]
    n = min(muestras(dur_ms, fs), x.size)
    centro = int(tiempos[i] * fs)
    inicio = int(np.clip(centro - n // 2, 0, max(0, x.size - n)))
    return x[inicio : inicio + n], inicio / fs


def espectrograma(x, fs, dur_ms=25.0, salto_ms=5.0, ventana="hamming", n_fft=None):
    """Espectrograma de potencia en dB.

    Con tramas cortas (~5 ms) se obtiene un espectrograma de *banda ancha*,
    que resuelve bien en el tiempo y muestra los formantes; con tramas largas
    (~40 ms) uno de *banda estrecha*, que resuelve los armónicos del pitch.

    Devuelve ``(S_db, frecuencias, tiempos)`` con ``S_db`` de forma
    ``(n_frecuencias, n_tramas)``.
    """
    tramas, tiempos = enventanar(x, fs, dur_ms, salto_ms, ventana)
    if n_fft is None:
        n_fft = max(512, 2 ** int(np.ceil(np.log2(tramas.shape[1]))))
    potencia = np.abs(np.fft.rfft(tramas, n_fft, axis=1)) ** 2
    s_db = 10 * np.log10(potencia + 1e-12)
    frecuencias = np.fft.rfftfreq(n_fft, 1 / fs)
    return s_db.T, frecuencias, tiempos

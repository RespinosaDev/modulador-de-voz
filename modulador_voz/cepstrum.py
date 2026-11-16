"""Análisis cepstral.

El cepstrum real es la transformada inversa del logaritmo del módulo del
espectro::

    c[q] = IFFT( log |X(f)| )

Como en el modelo fuente-filtro el espectro es el *producto* de la
excitación y el tracto vocal, el logaritmo lo convierte en una *suma*: en el
cepstrum el tracto vocal ocupa las quefrencias bajas y la excitación las
altas (con un pico en el periodo fundamental). Separarlas con una ventana
en quefrencia se llama *liftering*.
"""

from __future__ import annotations

import numpy as np

from .pitch import refinar_pico


def _n_fft(n: int) -> int:
    return max(512, 2 ** int(np.ceil(np.log2(n))))


def cepstrum_real(trama: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Cepstrum real de una trama (ya enventanada)."""
    n_fft = n_fft or _n_fft(len(trama))
    log_mag = np.log(np.abs(np.fft.rfft(trama, n_fft)) + 1e-10)
    return np.fft.irfft(log_mag, n_fft)


def liftering(c: np.ndarray, n_coef: int):
    """Separa el cepstrum en parte de baja y de alta quefrencia.

    Devuelve ``(c_bajo, c_alto)``: ``c_bajo`` conserva los ``n_coef``
    primeros coeficientes (y sus simétricos) y representa la envolvente
    (tracto vocal); ``c_alto`` es el resto y representa la excitación.
    """
    c = np.asarray(c, dtype=float)
    bajo = np.zeros_like(c)
    bajo[:n_coef] = c[:n_coef]
    if n_coef > 1:
        bajo[-(n_coef - 1) :] = c[-(n_coef - 1) :]
    return bajo, c - bajo


def espectro_desde_cepstrum(c: np.ndarray) -> np.ndarray:
    """Espectro en dB correspondiente a un cepstrum (o a una parte de él)."""
    return 20 / np.log(10) * np.real(np.fft.rfft(c))


def coeficientes_lifter(fs: int, quefrencia_ms: float = 2.0) -> int:
    """Número de coeficientes de un lifter de ``quefrencia_ms`` milisegundos."""
    return max(2, int(fs * quefrencia_ms / 1000))


def pitch_cepstral(tramas, fs, f0_min=60.0, f0_max=400.0, umbral=0.06):
    """F0 de cada trama a partir del pico cepstral (0 si no hay pico claro)."""
    n_fft = _n_fft(tramas.shape[1])
    log_mag = np.log(np.abs(np.fft.rfft(tramas, n_fft, axis=1)) + 1e-10)
    c = np.fft.irfft(log_mag, n_fft, axis=1)

    q_min = max(2, int(fs / f0_max))
    q_max = min(int(np.ceil(fs / f0_min)), n_fft // 2 - 2)
    f0 = np.zeros(len(tramas))
    if q_min >= q_max:
        return f0
    for i, fila in enumerate(c):
        k = q_min + int(np.argmax(fila[q_min : q_max + 1]))
        if fila[k] >= umbral:
            f0[i] = fs / refinar_pico(fila, k)
    return f0

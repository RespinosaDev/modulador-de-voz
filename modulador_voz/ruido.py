"""Ruido: adición con SNR controlada y reducción por sustracción espectral."""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .analisis import muestras

TIPOS_RUIDO = ("blanco", "rosa")


def potencia(x: np.ndarray) -> float:
    """Potencia media de la señal."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x**2)) if x.size else 0.0


def snr_db(limpia: np.ndarray, ruidosa: np.ndarray) -> float:
    """Relación señal a ruido (dB) de ``ruidosa`` respecto a ``limpia``."""
    limpia = np.asarray(limpia, dtype=float)
    ruidosa = np.asarray(ruidosa, dtype=float)
    n = min(limpia.size, ruidosa.size)
    p_ruido = potencia(ruidosa[:n] - limpia[:n])
    if p_ruido == 0:
        return float("inf")
    return 10 * np.log10(potencia(limpia[:n]) / p_ruido)


def generar_ruido(n: int, tipo: str = "blanco", semilla: int | None = None) -> np.ndarray:
    """Ruido blanco (espectro plano) o rosa (densidad espectral 1/f)."""
    rng = np.random.default_rng(semilla)
    ruido = rng.standard_normal(n)
    if tipo == "rosa":
        espectro = np.fft.rfft(ruido)
        f = np.arange(espectro.size)
        f[0] = 1
        ruido = np.fft.irfft(espectro / np.sqrt(f), n)
    elif tipo != "blanco":
        raise ValueError(f"Tipo de ruido desconocido: {tipo!r}. Opciones: {', '.join(TIPOS_RUIDO)}")
    return ruido


def anadir_ruido(x, snr_objetivo_db, tipo="blanco", semilla=None):
    """Suma ruido a la señal con la SNR indicada. Devuelve ``(ruidosa, ruido)``."""
    x = np.asarray(x, dtype=float)
    ruido = generar_ruido(x.size, tipo, semilla)
    ruido *= np.sqrt(potencia(x) / (potencia(ruido) * 10 ** (snr_objetivo_db / 10)))
    return x + ruido, ruido


def sustraccion_espectral(x, fs, alfa=2.0, beta=0.02, dur_ms=32.0, segundos_ruido=None):
    """Reducción de ruido por sustracción espectral (Boll, 1979).

    1. Se estima el espectro de potencia del ruido ``N(f)``: con los primeros
       ``segundos_ruido`` segundos si se indican o, si no, con el 10 % de las
       tramas de menor energía (normalmente pausas).
    2. En cada trama: ``|S|² = max(|X|² - alfa·|N|², beta·|X|²)``. El
       sobresustraer (``alfa > 1``) y el suelo espectral (``beta``) reducen el
       «ruido musical».
    3. Se reconstruye con la fase de la señal ruidosa (ISTFT).
    """
    x = np.asarray(x, dtype=float)
    nperseg = muestras(dur_ms, fs)
    _, tiempos, espectro = stft(x, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    potencia_tramas = np.abs(espectro) ** 2

    if segundos_ruido:
        seleccion = tiempos <= segundos_ruido
    else:
        energia_trama = potencia_tramas.sum(axis=0)
        seleccion = energia_trama <= np.percentile(energia_trama, 10)
    if not np.any(seleccion):
        seleccion = np.zeros(tiempos.size, dtype=bool)
        seleccion[0] = True
    ruido = potencia_tramas[:, seleccion].mean(axis=1, keepdims=True)

    limpia = np.maximum(potencia_tramas - alfa * ruido, beta * potencia_tramas)
    espectro_limpio = np.sqrt(limpia) * np.exp(1j * np.angle(espectro))
    _, y = istft(espectro_limpio, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    y = y[: x.size]
    return np.pad(y, (0, x.size - y.size))

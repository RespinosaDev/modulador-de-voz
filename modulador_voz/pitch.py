"""Estimación de la frecuencia fundamental (pitch, F0).

Se implementan tres métodos clásicos sobre tramas:

- Autocorrelación (con recorte central): el primer máximo de la
  autocorrelación fuera del origen aparece en el periodo fundamental.
- AMDF (Average Magnitude Difference Function): el mínimo de la diferencia
  media entre la señal y su versión desplazada aparece en el periodo.
- Cepstrum: la excitación periódica produce un pico en la quefrencia igual
  al periodo (se implementa en :mod:`modulador_voz.cepstrum`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from .analisis import SONORO, clasificar_tramas, crear_ventana, enventanar
from .audio_io import remuestrear

METODOS_PITCH = ("autocorrelacion", "amdf", "cepstrum")

#: Por encima de esta fs se remuestrea antes de estimar el pitch (más rápido
#: y suficiente, ya que F0 < 500 Hz).
_FS_PITCH = 16000


def refinar_pico(y: np.ndarray, k: int) -> float:
    """Interpolación parabólica alrededor de ``y[k]`` para precisión inferior a una muestra."""
    if 0 < k < len(y) - 1:
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if den != 0:
            return k + 0.5 * (y[k - 1] - y[k + 1]) / den
    return float(k)


def recorte_central(tramas: np.ndarray, nivel: float = 0.3) -> np.ndarray:
    """Recorte central de Sondhi: anula las muestras por debajo de un umbral.

    Atenúa la influencia de los formantes y realza la periodicidad.
    """
    umbral = nivel * np.max(np.abs(tramas), axis=1, keepdims=True)
    return np.where(
        tramas > umbral,
        tramas - umbral,
        np.where(tramas < -umbral, tramas + umbral, 0.0),
    )


def autocorrelacion_tramas(tramas: np.ndarray) -> np.ndarray:
    """Autocorrelación de cada trama (vía FFT), lags 0..N-1."""
    n = tramas.shape[1]
    n_fft = 2 ** int(np.ceil(np.log2(2 * n)))
    espectro = np.fft.rfft(tramas, n_fft, axis=1)
    return np.fft.irfft(np.abs(espectro) ** 2, n_fft, axis=1)[:, :n]


def _rango_lags(fs, f0_min, f0_max, n):
    lag_min = max(2, int(fs / f0_max))
    lag_max = min(int(np.ceil(fs / f0_min)), n - 2)
    return lag_min, lag_max


def pitch_autocorrelacion(tramas, fs, f0_min=60.0, f0_max=400.0, umbral=0.3, recorte=False, ventana=None):
    """F0 de cada trama por autocorrelación normalizada (0 si no es periódica).

    Si se indica la ``ventana`` con la que se enventanaron las tramas, la
    autocorrelación se divide por la de la ventana (Boersma, 1993) para
    compensar la atenuación que esta produce en los lags grandes. Con esa
    corrección el recorte central es opcional (``recorte=True``).
    """
    t = tramas - tramas.mean(axis=1, keepdims=True)
    if recorte:
        t = recorte_central(t)
    r = autocorrelacion_tramas(t)
    r0 = r[:, :1]
    r = np.divide(r, r0, out=np.zeros_like(r), where=r0 > 0)
    if ventana is not None:
        r_w = autocorrelacion_tramas(np.asarray(ventana, dtype=float)[None, :])[0]
        r_w = r_w / r_w[0]
        r = r / np.maximum(r_w, 0.1)[None, :]

    f0 = np.zeros(len(r))
    lag_min, lag_max = _rango_lags(fs, f0_min, f0_max, r.shape[1])
    if lag_min >= lag_max:
        return f0
    for i, fila in enumerate(r):
        zona = fila[lag_min : lag_max + 1]
        maximo = zona.max()
        if maximo < umbral:
            continue
        # Primer máximo local cercano al global: evita elegir 2T, 3T... (errores de octava).
        locales = np.flatnonzero((zona[1:-1] >= zona[:-2]) & (zona[1:-1] >= zona[2:]) & (zona[1:-1] >= 0.9 * maximo)) + 1
        k = lag_min + (int(locales[0]) if locales.size else int(np.argmax(zona)))
        f0[i] = fs / refinar_pico(fila, k)
    return f0


def pitch_amdf(tramas, fs, f0_min=60.0, f0_max=400.0, umbral=0.45):
    """F0 de cada trama con la función AMDF (0 si no es periódica)."""
    n = tramas.shape[1]
    lag_min, lag_max = _rango_lags(fs, f0_min, f0_max, n)
    f0 = np.zeros(len(tramas))
    if lag_min >= lag_max:
        return f0
    lags = np.arange(lag_min, lag_max + 1)
    d = np.stack([np.mean(np.abs(tramas[:, : n - k] - tramas[:, k:]), axis=1) for k in lags], axis=1)

    for i, fila in enumerate(d):
        minimo, maximo = fila.min(), fila.max()
        if maximo <= 0 or minimo / (fila.mean() + 1e-12) > umbral:
            continue
        # Primer valle cercano al mínimo global (evita elegir múltiplos del periodo).
        k = int(np.flatnonzero(fila <= minimo + 0.15 * (maximo - minimo))[0])
        while k + 1 < len(fila) and fila[k + 1] < fila[k]:
            k += 1
        f0[i] = fs / (lag_min + refinar_pico(-fila, k))
    return f0


def estimar_pitch(
    x,
    fs,
    metodo="autocorrelacion",
    dur_ms=40.0,
    salto_ms=10.0,
    f0_min=60.0,
    f0_max=400.0,
    solo_sonoras=True,
):
    """Contorno de pitch de la señal.

    Devuelve ``(f0, tiempos)``: F0 en Hz por trama (0 en tramas no sonoras) y
    el instante central de cada trama en segundos.
    """
    if metodo not in METODOS_PITCH:
        raise ValueError(f"Método desconocido: {metodo!r}. Opciones: {', '.join(METODOS_PITCH)}")
    x = np.asarray(x, dtype=float)
    if fs > _FS_PITCH:
        x = remuestrear(x, fs, _FS_PITCH)
        fs = _FS_PITCH

    if metodo == "autocorrelacion":
        tramas, tiempos = enventanar(x, fs, dur_ms, salto_ms, "hamming")
        f0 = pitch_autocorrelacion(
            tramas, fs, f0_min, f0_max, ventana=crear_ventana("hamming", tramas.shape[1])
        )
    elif metodo == "amdf":
        tramas, tiempos = enventanar(x, fs, dur_ms, salto_ms, "rectangular")
        f0 = pitch_amdf(tramas, fs, f0_min, f0_max)
    else:
        from .cepstrum import pitch_cepstral

        tramas, tiempos = enventanar(x, fs, dur_ms, salto_ms, "hamming")
        f0 = pitch_cepstral(tramas, fs, f0_min, f0_max)

    if solo_sonoras:
        clases, *_ = clasificar_tramas(x, fs, dur_ms, salto_ms)
        m = min(len(clases), len(f0))
        f0[:m][clases[:m] != SONORO] = 0.0

    # Filtro de mediana para eliminar saltos de octava aislados.
    if len(f0) >= 5:
        suavizado = medfilt(f0, 5)
        f0 = np.where((f0 > 0) & (suavizado > 0), suavizado, f0)
    return f0, tiempos

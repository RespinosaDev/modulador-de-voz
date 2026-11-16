"""Modificación de la voz: cambio de tono (semitonos) y de velocidad.

- **Semitonos**: subir ``n`` semitonos multiplica F0 por ``2^(n/12)``.
  Se implementa con TD-PSOLA o con phase vocoder + remuestreo, de modo que
  la duración no cambia.
- **Velocidad** (x0,5, x1,5, x2...): con phase vocoder se cambia la
  duración sin alterar el tono; con remuestreo simple cambian las dos cosas
  a la vez (efecto «cinta acelerada»).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import get_window, resample_poly

from .analisis import muestras
from .pitch import estimar_pitch

METODOS_TONO = ("psola", "vocoder")


def factor_semitonos(semitonos: float) -> float:
    """Factor de frecuencia correspondiente a ``semitonos`` (escala temperada)."""
    return 2.0 ** (semitonos / 12.0)


def cambiar_longitud(x: np.ndarray, n_destino: int) -> np.ndarray:
    """Remuestrea la señal para que tenga exactamente ``n_destino`` muestras."""
    x = np.asarray(x, dtype=float)
    if n_destino == x.size:
        return x.copy()
    r = Fraction(n_destino, x.size).limit_denominator(500)
    y = resample_poly(x, r.numerator, r.denominator)
    if y.size >= n_destino:
        return y[:n_destino]
    return np.pad(y, (0, n_destino - y.size))


# --------------------------------------------------------------------------
# TD-PSOLA
# --------------------------------------------------------------------------


def marcas_pitch(x, fs, f0=None, tiempos=None, periodo_sordo_ms=10.0):
    """Marcas de análisis síncronas con el pitch.

    En tramas sonoras se coloca una marca por periodo, ajustada al máximo
    local de la señal; en las sordas, marcas equiespaciadas.

    Devuelve ``(marcas, periodos, sonoras)``.
    """
    x = np.asarray(x, dtype=float)
    if f0 is None or tiempos is None:
        f0, tiempos = estimar_pitch(x, fs)
    n = x.size
    periodo_sordo = muestras(periodo_sordo_ms, fs)

    marcas, periodos, sonoras = [], [], []
    m = 0
    while m < n:
        j = min(int(np.searchsorted(tiempos, m / fs)), len(f0) - 1)
        if f0[j] > 0:
            periodo = max(2, int(round(fs / f0[j])))
            inicio = max(0, m - periodo // 4)
            fin = min(n, m + periodo // 4 + 1)
            if marcas:
                inicio = max(inicio, marcas[-1] + periodo // 2)
            if fin > inicio:
                m = inicio + int(np.argmax(x[inicio:fin]))
            marcas.append(m)
            periodos.append(periodo)
            sonoras.append(True)
            m += periodo
        else:
            marcas.append(m)
            periodos.append(periodo_sordo)
            sonoras.append(False)
            m += periodo_sordo
    return np.array(marcas), np.array(periodos), np.array(sonoras)


def psola(x, fs, factor, f0=None, tiempos=None):
    """Cambio de tono por TD-PSOLA (*Pitch-Synchronous Overlap-Add*).

    Se extraen segmentos de dos periodos centrados en cada marca de pitch
    (ventana de Hann) y se vuelven a sumar con una separación de
    ``periodo / factor``. Así cambia F0 sin cambiar la duración ni los
    formantes. Las zonas sordas se copian sin modificar.
    """
    x = np.asarray(x, dtype=float)
    if abs(factor - 1.0) < 1e-9 or x.size == 0:
        return x.copy()
    marcas, periodos, sonoras = marcas_pitch(x, fs, f0, tiempos)
    n = x.size
    relleno = int(periodos.max()) + 1
    xp = np.pad(x, (relleno, relleno))
    salida = np.zeros(n + 2 * relleno)
    peso = np.zeros(n + 2 * relleno)
    ventanas = {}

    t_sintesis = float(marcas[0])
    while t_sintesis < n:
        k = int(np.searchsorted(marcas, t_sintesis))
        if k == len(marcas) or (k > 0 and t_sintesis - marcas[k - 1] < marcas[k] - t_sintesis):
            k -= 1
        periodo, marca = int(periodos[k]), int(marcas[k]) + relleno
        if periodo not in ventanas:
            ventanas[periodo] = np.hanning(2 * periodo + 1)
        w = ventanas[periodo]
        p = int(round(t_sintesis)) + relleno
        salida[p - periodo : p + periodo + 1] += xp[marca - periodo : marca + periodo + 1] * w
        peso[p - periodo : p + periodo + 1] += w
        t_sintesis += periodo / factor if sonoras[k] else periodo

    salida = salida[relleno : relleno + n]
    return salida / np.maximum(peso[relleno : relleno + n], 1.0)


# --------------------------------------------------------------------------
# Phase vocoder
# --------------------------------------------------------------------------


def _tamano_fft(fs: int) -> int:
    return 2 ** int(np.ceil(np.log2(0.046 * fs)))  # ~46 ms


def phase_vocoder(x, tasa, fs):
    """Cambio de duración sin cambio de tono mediante *phase vocoder*.

    ``tasa > 1`` acelera (la salida dura ``1/tasa`` veces la original) y
    ``tasa < 1`` ralentiza. Se calcula la STFT, se interpolan los módulos en
    los nuevos instantes y se reconstruye la fase acumulando la frecuencia
    instantánea de cada canal, para mantener la coherencia de fase.
    """
    x = np.asarray(x, dtype=float)
    if abs(tasa - 1.0) < 1e-9 or x.size == 0:
        return x.copy()
    n_fft = _tamano_fft(fs)
    salto = n_fft // 4
    w = get_window("hann", n_fft)

    xp = np.pad(x, (n_fft // 2, n_fft))
    n_tramas = 1 + (xp.size - n_fft) // salto
    indices = salto * np.arange(n_tramas)[:, None] + np.arange(n_fft)[None, :]
    stft = np.fft.rfft(xp[indices] * w, axis=1)

    instantes = np.arange(0, n_tramas - 1, tasa)
    avance = 2 * np.pi * salto * np.arange(stft.shape[1]) / n_fft
    fase = np.angle(stft[0])
    salida_stft = np.empty((len(instantes), stft.shape[1]), dtype=complex)
    for j, t in enumerate(instantes):
        i = int(t)
        frac = t - i
        c0, c1 = stft[i], stft[i + 1]
        modulo = (1 - frac) * np.abs(c0) + frac * np.abs(c1)
        salida_stft[j] = modulo * np.exp(1j * fase)
        delta = np.angle(c1) - np.angle(c0) - avance
        delta -= 2 * np.pi * np.round(delta / (2 * np.pi))
        fase = fase + avance + delta

    tramas = np.fft.irfft(salida_stft, n_fft, axis=1) * w
    y = np.zeros(n_fft + salto * (len(instantes) - 1))
    norma = np.zeros_like(y)
    for j, trama in enumerate(tramas):
        y[j * salto : j * salto + n_fft] += trama
        norma[j * salto : j * salto + n_fft] += w**2
    y = np.divide(y, norma, out=np.zeros_like(y), where=norma > 1e-3)

    n_destino = int(round(x.size / tasa))
    y = y[n_fft // 2 : n_fft // 2 + n_destino]
    return np.pad(y, (0, max(0, n_destino - y.size)))


# --------------------------------------------------------------------------
# Funciones de alto nivel
# --------------------------------------------------------------------------


def cambiar_semitonos(x, fs, semitonos, metodo="psola"):
    """Sube o baja el tono ``semitonos`` semitonos manteniendo la duración."""
    x = np.asarray(x, dtype=float)
    if semitonos == 0:
        return x.copy()
    factor = factor_semitonos(semitonos)
    if metodo == "psola":
        return psola(x, fs, factor)
    if metodo == "vocoder":
        # Se alarga la señal por el factor sin cambiar el tono y después se
        # remuestrea a la longitud original, lo que multiplica F0 por el factor.
        estirada = phase_vocoder(x, 1.0 / factor, fs)
        return cambiar_longitud(estirada, x.size)
    raise ValueError(f"Método desconocido: {metodo!r}. Opciones: {', '.join(METODOS_TONO)}")


def cambiar_velocidad(x, fs, velocidad, mantener_tono=True):
    """Cambia la velocidad de reproducción (x0,5, x1,5, x2...).

    Con ``mantener_tono`` se usa el phase vocoder; si no, un remuestreo
    simple que también cambia el tono, como al acelerar una cinta.
    """
    x = np.asarray(x, dtype=float)
    if velocidad <= 0:
        raise ValueError("La velocidad debe ser positiva.")
    if velocidad == 1:
        return x.copy()
    if mantener_tono:
        return phase_vocoder(x, velocidad, fs)
    return cambiar_longitud(x, int(round(x.size / velocidad)))

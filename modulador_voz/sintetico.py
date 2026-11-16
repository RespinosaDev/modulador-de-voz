"""Voz sintética para pruebas y demostraciones.

Genera vocales con el modelo fuente-filtro: un tren de pulsos glotales de
Rosenberg, derivado (radiación labial) y filtrado por resonadores que
imitan los tres primeros formantes. Al conocer la fuente y el filtro reales
se pueden comprobar las estimaciones de pitch, formantes y flujo glotal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal.windows import tukey

#: Formantes aproximados (frecuencia Hz, ancho de banda Hz) de las vocales del español.
VOCALES = {
    "a": ((730, 80), (1090, 90), (2440, 120)),
    "e": ((530, 60), (1840, 100), (2480, 120)),
    "i": ((270, 50), (2290, 100), (3010, 150)),
    "o": ((570, 70), (840, 80), (2410, 120)),
    "u": ((300, 50), (870, 80), (2240, 120)),
}


def pulso_rosenberg(periodo: int, apertura: float = 0.4, cierre: float = 0.16) -> np.ndarray:
    """Un periodo del pulso glotal de Rosenberg (fases de apertura y cierre)."""
    tp = max(1, int(round(apertura * periodo)))
    tn = max(1, int(round(cierre * periodo)))
    n = np.arange(periodo)
    g = np.zeros(periodo)
    abre = n <= tp
    g[abre] = 0.5 * (1 - np.cos(np.pi * n[abre] / tp))
    cierra = (n > tp) & (n <= tp + tn)
    g[cierra] = np.cos(np.pi * (n[cierra] - tp) / (2 * tn))
    return g


def flujo_glotal(fs, duracion, f0_inicial, f0_final=None, jitter=0.0, semilla=0):
    """Tren de pulsos glotales con F0 que varía linealmente."""
    f0_final = f0_final or f0_inicial
    rng = np.random.default_rng(semilla)
    total = int(duracion * fs)
    g = np.zeros(total)
    pos = 0
    while pos < total:
        f0 = f0_inicial + (f0_final - f0_inicial) * pos / total
        periodo = max(2, int(round(fs / f0 * (1 + jitter * rng.standard_normal()))))
        pulso = pulso_rosenberg(periodo)
        fin = min(total, pos + periodo)
        g[pos:fin] = pulso[: fin - pos]
        pos += periodo
    return g


def tracto_vocal(formantes, fs) -> np.ndarray:
    """Polinomio ``A(z)`` de una cascada de resonadores de dos polos."""
    a = np.array([1.0])
    for frecuencia, ancho in formantes:
        r = np.exp(-np.pi * ancho / fs)
        a = np.convolve(a, [1.0, -2 * r * np.cos(2 * np.pi * frecuencia / fs), r * r])
    return a


def vocal_sintetica(vocal="a", fs=16000, duracion=0.5, f0_inicial=120.0, f0_final=None, jitter=0.0, semilla=0):
    """Vocal sintética. Devuelve ``(voz, flujo_glotal)``."""
    g = flujo_glotal(fs, duracion, f0_inicial, f0_final, jitter, semilla)
    derivada = np.diff(g, prepend=0.0)  # radiación labial ~ derivada
    voz = lfilter([1.0], tracto_vocal(VOCALES[vocal], fs), derivada)
    return 0.8 * voz / np.max(np.abs(voz)), g


def frase_sintetica(fs=16000, semilla=0):
    """Secuencia «a - s - i - u» con pausas y entonación variable.

    Incluye silencios, una fricativa sorda y tres vocales sonoras, de modo
    que sirve para probar todos los análisis. Devuelve ``(x, fs)``.
    """
    rng = np.random.default_rng(semilla)

    def silencio(segundos):
        return 0.002 * rng.standard_normal(int(segundos * fs))

    def fricativa(segundos):
        b, a = butter(4, 3000 / (fs / 2), btype="high")
        ruido = lfilter(b, a, rng.standard_normal(int(segundos * fs)))
        return 0.25 * ruido / np.max(np.abs(ruido)) * tukey(ruido.size, 0.5)

    def vocal(v, segundos, f0_ini, f0_fin):
        voz, _ = vocal_sintetica(v, fs, segundos, f0_ini, f0_fin, jitter=0.005, semilla=int(rng.integers(1000)))
        return voz * tukey(voz.size, 0.15)

    partes = [
        silencio(0.25),
        vocal("a", 0.45, 110, 140),
        silencio(0.08),
        fricativa(0.15),
        vocal("i", 0.45, 140, 125),
        silencio(0.10),
        vocal("u", 0.40, 125, 100),
        silencio(0.25),
    ]
    x = np.concatenate(partes)
    return 0.9 * x / np.max(np.abs(x)), fs

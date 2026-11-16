"""Predicción lineal (LPC) y modelo fuente-filtro de la voz.

En el modelo fuente-filtro, la voz es la salida de un filtro (el tracto
vocal) excitado por una fuente (el flujo de aire glotal en sonidos sonoros o
ruido en los sordos). El tracto vocal se modela como un filtro todo polos::

    H(z) = G / A(z),   A(z) = 1 + a1 z^-1 + ... + ap z^-p

Los coeficientes ``a_k`` se obtienen minimizando el error de predicción
lineal, lo que lleva a las ecuaciones normales de Yule-Walker, que se
resuelven con el algoritmo de Levinson-Durbin.

Con ``A(z)`` se puede:

- Dibujar la envolvente espectral ``G / |A(e^jw)|`` y obtener los formantes
  a partir de los polos.
- Aplicar el *filtrado inverso* ``e[n] = A(z) x[n]`` para separar la
  excitación (residuo) del tracto vocal.
- Estimar el flujo glotal con IAIF (filtrado inverso adaptativo iterativo).
- Sintetizar voz con otra excitación (efectos robot y susurro).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt, freqz, get_window, lfilter

from .analisis import crear_ventana, muestras, preenfasis


def orden_recomendado(fs: int) -> int:
    """Regla práctica: un par de polos por kHz de ancho de banda + 2."""
    return int(fs / 1000) + 2


def autocorrelacion(x: np.ndarray, orden: int) -> np.ndarray:
    """Autocorrelación ``r[0..orden]`` (sin normalizar)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    return np.array([np.dot(x[: n - k], x[k:]) for k in range(orden + 1)])


def levinson_durbin(r: np.ndarray, orden: int):
    """Resuelve las ecuaciones de Yule-Walker de forma recursiva.

    Devuelve ``(a, error, k)``:

    - ``a``: coeficientes ``[1, a1, ..., ap]`` del polinomio ``A(z)``.
    - ``error``: energía del error de predicción.
    - ``k``: coeficientes de reflexión (PARCOR).
    """
    a = np.zeros(orden + 1)
    a[0] = 1.0
    reflexion = np.zeros(orden)
    error = float(r[0])
    if error <= 0:
        return a, 0.0, reflexion
    for i in range(1, orden + 1):
        acumulado = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        k = -acumulado / error
        anterior = a.copy()
        a[1:i] = anterior[1:i] + k * anterior[i - 1 : 0 : -1]
        a[i] = k
        reflexion[i - 1] = k
        error *= 1.0 - k * k
        if error <= 0:
            break
    return a, max(error, 0.0), reflexion


def lpc(trama: np.ndarray, orden: int):
    """Coeficientes LPC de una trama (ya enventanada) por autocorrelación.

    Devuelve ``(a, ganancia)`` con ``ganancia = sqrt(energía del error)``.
    """
    r = autocorrelacion(trama, orden)
    if r[0] <= 0:
        a = np.zeros(orden + 1)
        a[0] = 1.0
        return a, 0.0
    r[0] *= 1.0 + 1e-9  # pequeña regularización (ruido blanco de -90 dB)
    a, error, _ = levinson_durbin(r, orden)
    return a, float(np.sqrt(error))


def envolvente_lpc(a, ganancia, fs, n_puntos=1024):
    """Respuesta en frecuencia ``G / A(e^jw)`` en dB. Devuelve ``(f, H_db)``."""
    f, h = freqz([ganancia], a, worN=n_puntos, fs=fs)
    return f, 20 * np.log10(np.abs(h) + 1e-12)


def formantes(a, fs, n=4, f_min=90.0, ancho_banda_max=400.0):
    """Frecuencias y anchos de banda de los formantes a partir de los polos.

    Cada par de polos complejos conjugados ``r·e^{±jθ}`` es una resonancia con
    frecuencia ``θ·fs/2π`` y ancho de banda ``-ln(r)·fs/π``. Se descartan
    los polos reales, los de muy baja frecuencia y los muy amortiguados.

    Devuelve una lista de tuplas ``(frecuencia_Hz, ancho_banda_Hz)``.
    """
    polos = np.roots(a)
    polos = polos[np.imag(polos) > 0]
    frecuencias = np.angle(polos) * fs / (2 * np.pi)
    anchos = -np.log(np.abs(polos)) * fs / np.pi
    validos = (frecuencias > f_min) & (anchos < ancho_banda_max)
    orden = np.argsort(frecuencias[validos])
    return list(zip(frecuencias[validos][orden][:n], anchos[validos][orden][:n]))


def lpc_tramas(x, fs, orden=None, dur_ms=30.0, salto_ms=10.0, ventana="hamming", coef_preenfasis=0.97):
    """Coeficientes LPC de todas las tramas de la señal.

    Devuelve ``(A, ganancias, tiempos)`` con ``A`` de forma ``(n_tramas, orden+1)``.
    """
    from .analisis import enventanar

    orden = orden or orden_recomendado(fs)
    xp = preenfasis(x, coef_preenfasis) if coef_preenfasis else np.asarray(x, dtype=float)
    tramas, tiempos = enventanar(xp, fs, dur_ms, salto_ms, ventana)
    resultados = [lpc(t, orden) for t in tramas]
    coeficientes = np.array([r[0] for r in resultados])
    ganancias = np.array([r[1] for r in resultados])
    return coeficientes, ganancias, tiempos


def residuo_lpc(x, fs, orden=None, dur_ms=30.0, coef_preenfasis=0.97):
    """Filtrado inverso trama a trama: ``e[n] = A(z) x[n]``.

    El residuo (error de predicción) aproxima la excitación: un tren de
    impulsos en los instantes de cierre glotal en tramas sonoras y ruido en
    las sordas. Las tramas se solapan al 50 % y se recombinan por
    solapamiento y suma (OLA) con ventana de Hann.
    """
    x = np.asarray(x, dtype=float)
    orden = orden or orden_recomendado(fs)
    n = muestras(dur_ms, fs)
    salto = n // 2
    xp = preenfasis(x, coef_preenfasis) if coef_preenfasis else x
    w_analisis = crear_ventana("hamming", n)
    w_sintesis = get_window("hann", n)  # periódica: suma constante al 50 %

    relleno = np.pad(xp, (salto, n))
    salida = np.zeros_like(relleno)
    for inicio in range(0, relleno.size - n + 1, salto):
        segmento = relleno[inicio : inicio + n]
        a, _ = lpc(segmento * w_analisis, orden)
        salida[inicio : inicio + n] += lfilter(a, [1.0], segmento) * w_sintesis
    return salida[salto : salto + x.size]


def _integrar(x, coef=0.99):
    """Integrador con pérdidas ``1 / (1 - coef z^-1)``."""
    return lfilter([1.0], [1.0, -coef], x)


def iaif(segmento, fs, orden_vt=None, orden_glotal=4, coef_integrador=0.99):
    """Estimación del flujo glotal por IAIF (Alku, 1992).

    *Iterative Adaptive Inverse Filtering*: separa la contribución del tracto
    vocal y de la fuente glotal mediante dos iteraciones de filtrado inverso.

    1. Se modela la inclinación espectral glotal con LPC de orden 1 y se
       elimina de la voz.
    2. Se estima el tracto vocal (LPC de orden ``orden_vt``) y se filtra
       inversamente la voz; integrando se obtiene una primera estimación del
       flujo glotal.
    3. Se repite modelando la fuente glotal con orden ``orden_glotal`` para
       obtener un modelo del tracto vocal más limpio.

    Pensado para un segmento sonoro corto (30-100 ms). Devuelve
    ``(flujo_glotal, derivada_flujo_glotal, coeficientes_tracto_vocal)``.
    """
    x = np.asarray(segmento, dtype=float)
    orden_vt = orden_vt or orden_recomendado(fs)

    # Paso alto para eliminar componentes de muy baja frecuencia.
    b, a = butter(2, 40 / (fs / 2), btype="high")
    if x.size > 3 * max(len(a), len(b)):
        x = filtfilt(b, a, x)
    w = np.hanning(x.size)

    def integrar(s):
        return _integrar(s, coef_integrador)

    # Iteración 1
    g1, _ = lpc(x * w, 1)
    y1 = lfilter(g1, [1.0], x)
    vt1, _ = lpc(y1 * w, orden_vt)
    flujo1 = integrar(lfilter(vt1, [1.0], x))

    # Iteración 2
    g2, _ = lpc(flujo1 * w, orden_glotal)
    y2 = integrar(lfilter(g2, [1.0], x))
    vt2, _ = lpc(y2 * w, orden_vt)
    derivada = lfilter(vt2, [1.0], x)
    flujo = integrar(derivada)

    # Eliminar la deriva lineal que introduce el integrador.
    flujo = detrend(flujo, type="linear")
    return flujo, derivada, vt2


def _excitacion(n, fs, modo, x=None, f0_robot=110.0, semilla=0):
    """Señal de excitación para la síntesis LPC (potencia media unitaria)."""
    rng = np.random.default_rng(semilla)
    if modo == "susurro":
        return rng.standard_normal(n)

    if modo == "robot":
        f0_muestra = np.full(n, float(f0_robot))
    else:  # "original": sigue el pitch de la señal
        from .pitch import estimar_pitch

        f0, tiempos = estimar_pitch(x, fs)
        indices = np.clip(np.searchsorted(tiempos * fs, np.arange(n)), 0, len(f0) - 1)
        f0_muestra = f0[indices]

    exc = np.zeros(n)
    bloque_sordo = muestras(5, fs)
    pos = 0
    while pos < n:
        f0 = f0_muestra[pos]
        if f0 > 0:
            periodo = max(1, int(round(fs / f0)))
            exc[pos] = np.sqrt(periodo)  # potencia media 1
            pos += periodo
        else:
            fin = min(n, pos + bloque_sordo)
            exc[pos:fin] = rng.standard_normal(fin - pos)
            pos = fin
    return exc


MODOS_SINTESIS = ("robot", "susurro", "original")


def sintesis_lpc(x, fs, modo="robot", f0_robot=110.0, orden=None, dur_ms=30.0, coef_preenfasis=0.97):
    """Resintetiza la voz con el filtro LPC de cada trama y otra excitación.

    - ``"robot"``: tren de impulsos de F0 constante (voz monótona, robótica).
    - ``"susurro"``: ruido blanco (voz susurrada, sin vibración glotal).
    - ``"original"``: tren de impulsos que sigue el pitch estimado y ruido en
      las tramas sordas (vocoder LPC clásico).
    """
    if modo not in MODOS_SINTESIS:
        raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {', '.join(MODOS_SINTESIS)}")
    x = np.asarray(x, dtype=float)
    orden = orden or orden_recomendado(fs)
    n = muestras(dur_ms, fs)
    salto = n // 2
    calentamiento = muestras(10, fs)  # estabiliza el filtro antes de cada trama

    xp = preenfasis(x, coef_preenfasis) if coef_preenfasis else x
    relleno = np.pad(xp, (0, n))
    exc = np.pad(_excitacion(relleno.size, fs, modo, x, f0_robot), (calentamiento, 0))
    w_analisis = crear_ventana("hamming", n)
    w_sintesis = get_window("hann", n)

    salida = np.zeros(relleno.size)
    for inicio in range(0, x.size, salto):
        a, ganancia = lpc(relleno[inicio : inicio + n] * w_analisis, orden)
        if ganancia == 0:
            continue
        e = exc[inicio : inicio + calentamiento + n]
        s = lfilter([ganancia / np.sqrt(n)], a, e)[calentamiento:]
        salida[inicio : inicio + n] += s * w_sintesis

    salida = salida[: x.size]
    if coef_preenfasis:
        salida = lfilter([1.0], [1.0, -coef_preenfasis], salida)  # deénfasis
    pico_original = np.max(np.abs(x)) if x.size else 0.0
    pico = np.max(np.abs(salida)) if salida.size else 0.0
    return salida * (pico_original / pico) if pico > 0 else salida

"""Gráficas de análisis (matplotlib).

Cada función recibe una ``Figure`` vacía y la rellena, de forma que se
pueden usar tanto en la interfaz (Tkinter) como en scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Patch
from scipy.signal import freqz

from . import analisis as an
from . import cepstrum as cp
from . import lpc as lp
from .pitch import estimar_pitch

COLOR_ORIGINAL = "#1f77b4"
COLOR_PROCESADA = "#d62728"
COLOR_SONORA = "#2ca02c"
COLOR_SORDA = "#ff7f0e"


@dataclass
class ParametrosAnalisis:
    """Parámetros comunes a las gráficas de análisis."""

    dur_ms: float = 30.0
    salto_ms: float = 10.0
    ventana: str = "hamming"
    orden_lpc: int | None = None  # None: fs/1000 + 2
    preenfasis: bool = True
    coef_preenfasis: float = 0.97
    metodo_pitch: str = "autocorrelacion"

    def orden(self, fs: int) -> int:
        return self.orden_lpc or lp.orden_recomendado(fs)

    def coef(self) -> float:
        return self.coef_preenfasis if self.preenfasis else 0.0


def _tiempo(x, fs, inicio=0.0):
    return inicio + np.arange(len(x)) / fs


def _dibujar_espectrograma(ax, x, fs, dur_ms, salto_ms, ventana="hamming", rango_db=70, f_max=None):
    s_db, f, t = an.espectrograma(x, fs, dur_ms, salto_ms, ventana)
    techo = s_db.max()
    ax.imshow(
        s_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=techo - rango_db,
        vmax=techo,
        interpolation="nearest",
    )
    ax.set_ylim(0, f_max or min(fs / 2, 8000))
    ax.set_ylabel("Frecuencia (Hz)")


def _espectro_db(segmento, fs, ventana="hamming", n_fft=None):
    n_fft = n_fft or max(1024, 2 ** int(np.ceil(np.log2(len(segmento)))))
    w = an.crear_ventana(ventana, len(segmento))
    espectro = 20 * np.log10(np.abs(np.fft.rfft(segmento * w, n_fft)) + 1e-12)
    return np.fft.rfftfreq(n_fft, 1 / fs), espectro


def _marcar_zonas(ax, clases, tiempos, salto_s):
    """Sombrea en el eje las zonas sonoras y sordas."""
    colores = {an.SONORO: COLOR_SONORA, an.SORDO: COLOR_SORDA}
    inicio = 0
    for i in range(1, len(clases) + 1):
        if i == len(clases) or clases[i] != clases[inicio]:
            color = colores.get(int(clases[inicio]))
            if color:
                ax.axvspan(tiempos[inicio] - salto_s / 2, tiempos[i - 1] + salto_s / 2, color=color, alpha=0.18, lw=0)
            inicio = i


# --------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------


def senal_y_espectrograma(fig, x, fs, p: ParametrosAnalisis):
    """Forma de onda y espectrogramas de banda ancha y banda estrecha."""
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
    ax1.plot(_tiempo(x, fs), x, lw=0.6, color=COLOR_ORIGINAL)
    ax1.set_title("Forma de onda")
    ax1.set_ylabel("Amplitud")
    xp = an.preenfasis(x, p.coef()) if p.preenfasis else x
    _dibujar_espectrograma(ax2, xp, fs, 5, 1, p.ventana)
    ax2.set_title("Espectrograma de banda ancha (ventana de 5 ms): formantes")
    _dibujar_espectrograma(ax3, xp, fs, 40, 5, p.ventana)
    ax3.set_title("Espectrograma de banda estrecha (ventana de 40 ms): armónicos del pitch")
    ax3.set_xlabel("Tiempo (s)")


def energia_y_zcr(fig, x, fs, p: ParametrosAnalisis):
    """Enventanado, energía, cruces por cero y decisión sonoro / sordo."""
    clases, t, e_rel, zcr = an.clasificar_tramas(x, fs, p.dur_ms, p.salto_ms)
    salto_s = p.salto_ms / 1000
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

    ax1.plot(_tiempo(x, fs), x, lw=0.6, color="0.25")
    _marcar_zonas(ax1, clases, t, salto_s)
    ax1.set_title(f"Señal segmentada en tramas de {p.dur_ms:g} ms (salto {p.salto_ms:g} ms)")
    ax1.set_ylabel("Amplitud")
    ax1.legend(
        handles=[Patch(color=COLOR_SONORA, alpha=0.4, label="Sonora"), Patch(color=COLOR_SORDA, alpha=0.4, label="Sorda")],
        loc="upper right",
        fontsize=8,
    )

    ax2.plot(t, e_rel, color=COLOR_ORIGINAL)
    ax2.axhline(-35, ls="--", color="0.4", lw=0.8, label="Umbral de silencio")
    ax2.set_ylabel("Energía (dB)")
    ax2.set_title("Energía a corto plazo (relativa al máximo)")
    ax2.legend(loc="lower right", fontsize=8)

    ax3.plot(t, zcr, color=COLOR_SORDA)
    ax3.axhline(3000, ls="--", color="0.4", lw=0.8, label="Umbral sonora / sorda")
    ax3.set_ylabel("Cruces / s")
    ax3.set_title("Tasa de cruces por cero (ZCR)")
    ax3.set_xlabel("Tiempo (s)")
    ax3.legend(loc="upper right", fontsize=8)


def pitch(fig, x, fs, p: ParametrosAnalisis):
    """Contorno de pitch sobre el espectrograma y comparación de métodos."""
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    _dibujar_espectrograma(ax1, x, fs, 40, 5, "hamming", f_max=min(fs / 2, 2000))
    f0, t = estimar_pitch(x, fs, p.metodo_pitch)
    sonoras = f0 > 0
    ax1.plot(t[sonoras], f0[sonoras], ".", color="cyan", ms=4, label=f"F0 ({p.metodo_pitch})")
    ax1.set_title("Espectrograma de banda estrecha y contorno de F0")
    ax1.legend(loc="upper right", fontsize=8)

    estilos = {"autocorrelacion": ("o", COLOR_ORIGINAL), "amdf": ("s", COLOR_SORDA), "cepstrum": ("^", COLOR_SONORA)}
    for metodo, (marcador, color) in estilos.items():
        f0_m, t_m = estimar_pitch(x, fs, metodo)
        validos = f0_m > 0
        media = np.median(f0_m[validos]) if np.any(validos) else 0
        ax2.plot(t_m[validos], f0_m[validos], marcador, ms=3, color=color, label=f"{metodo} (mediana {media:.0f} Hz)")
    ax2.set_title("Comparación de métodos de estimación de pitch")
    ax2.set_ylabel("F0 (Hz)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.legend(loc="upper right", fontsize=8)


def lpc_y_formantes(fig, x, fs, p: ParametrosAnalisis):
    """Espectro, envolvente LPC, formantes, polos y efecto del orden."""
    orden = p.orden(fs)
    segmento, t0 = an.segmento_sonoro(x, fs, max(p.dur_ms, 30))
    sp = an.preenfasis(segmento, p.coef()) if p.preenfasis else segmento
    w = an.crear_ventana(p.ventana, len(sp))
    a, g = lp.lpc(sp * w, orden)
    f_esp, espectro = _espectro_db(sp, fs, p.ventana)
    f_env, envolvente = lp.envolvente_lpc(a, g, fs)
    desplazamiento = espectro.max() - envolvente.max()
    lista_formantes = lp.formantes(a, fs)

    rejilla = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax1 = fig.add_subplot(rejilla[0, :])
    ax1.plot(f_esp, espectro, lw=0.7, color="0.55", label="Espectro (FFT)")
    ax1.plot(f_env, envolvente + desplazamiento, lw=2, color=COLOR_PROCESADA, label=f"Envolvente LPC (orden {orden})")
    for i, (f, bw) in enumerate(lista_formantes, start=1):
        ax1.axvline(f, ls="--", lw=0.8, color=COLOR_ORIGINAL)
        ax1.text(f, ax1.get_ylim()[1], f" F{i}\n {f:.0f} Hz", va="top", fontsize=8, color=COLOR_ORIGINAL)
    ax1.set_xlim(0, fs / 2)
    ax1.set_title(f"Segmento sonoro en t = {t0:.2f} s: espectro y envolvente LPC")
    ax1.set_xlabel("Frecuencia (Hz)")
    ax1.set_ylabel("dB")
    ax1.legend(loc="lower left", fontsize=8)

    ax2 = fig.add_subplot(rejilla[1, 0])
    circulo = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax2.plot(circulo.real, circulo.imag, color="0.6", lw=0.8)
    polos = np.roots(a)
    ax2.plot(polos.real, polos.imag, "x", color=COLOR_PROCESADA, ms=7)
    ax2.axhline(0, color="0.85", lw=0.5)
    ax2.axvline(0, color="0.85", lw=0.5)
    ax2.set_aspect("equal")
    ax2.set_title("Polos de 1/A(z) en el plano z")

    ax3 = fig.add_subplot(rejilla[1, 1])
    ax3.plot(f_esp, espectro, lw=0.5, color="0.75")
    for o in (4, 8, orden, 2 * orden):
        a_o, g_o = lp.lpc(sp * w, o)
        f_o, env_o = lp.envolvente_lpc(a_o, g_o, fs)
        ax3.plot(f_o, env_o + desplazamiento, lw=1.2, label=f"p = {o}")
    ax3.set_xlim(0, fs / 2)
    ax3.set_title("Efecto del orden del predictor")
    ax3.set_xlabel("Frecuencia (Hz)")
    ax3.legend(fontsize=7, loc="lower left")


def filtrado_inverso(fig, x, fs, p: ParametrosAnalisis):
    """Separación fuente-filtro: voz, residuo LPC y sus espectros."""
    orden = p.orden(fs)
    residuo = lp.residuo_lpc(x, fs, orden, p.dur_ms, p.coef())
    segmento, t0 = an.segmento_sonoro(x, fs, 60)
    i0 = int(round(t0 * fs))
    seg_residuo = residuo[i0 : i0 + len(segmento)]
    t = _tiempo(segmento, fs, t0) * 1000

    ax1, ax2, ax3 = fig.subplots(3, 1)
    ax1.plot(t, segmento, color=COLOR_ORIGINAL, lw=0.9)
    ax1.set_title("Voz s[n] (segmento sonoro)")
    ax1.set_ylabel("Amplitud")
    ax2.plot(t, seg_residuo, color=COLOR_PROCESADA, lw=0.9)
    ax2.set_title("Residuo e[n] = A(z)·s[n]: excitación (impulsos en los cierres glotales)")
    ax2.set_xlabel("Tiempo (ms)")
    ax2.set_ylabel("Amplitud")

    f1, e1 = _espectro_db(segmento, fs)
    f2, e2 = _espectro_db(seg_residuo, fs)
    ax3.plot(f1, e1 - e1.max(), color=COLOR_ORIGINAL, lw=0.8, label="Voz (con formantes)")
    ax3.plot(f2, e2 - e2.max() - 30, color=COLOR_PROCESADA, lw=0.8, label="Residuo (espectro plano, -30 dB)")
    ax3.set_xlim(0, fs / 2)
    ax3.set_title("Espectros: el filtrado inverso elimina la envolvente del tracto vocal")
    ax3.set_xlabel("Frecuencia (Hz)")
    ax3.set_ylabel("dB")
    ax3.legend(fontsize=8, loc="lower left")


def flujo_glotal(fig, x, fs, p: ParametrosAnalisis):
    """Separación tracto vocal / fuente glotal con IAIF."""
    orden = p.orden(fs)
    segmento, t0 = an.segmento_sonoro(x, fs, 120)
    flujo, derivada, vt = lp.iaif(segmento, fs, orden)
    # Se muestra la parte central para no ver los transitorios de los filtros.
    margen = min(an.muestras(30, fs), len(segmento) // 4)
    central = slice(margen, len(segmento) - margen)
    segmento, flujo, derivada = segmento[central], flujo[central], derivada[central]
    t = _tiempo(segmento, fs, t0 + margen / fs) * 1000

    rejilla = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    ax1 = fig.add_subplot(rejilla[0, 0])
    ax2 = fig.add_subplot(rejilla[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(rejilla[2, 0], sharex=ax1)
    ax1.plot(t, segmento, color=COLOR_ORIGINAL, lw=0.9)
    ax1.set_title("Voz s[n]")
    ax2.plot(t, derivada, color=COLOR_SORDA, lw=0.9)
    ax2.set_title("Derivada del flujo glotal (IAIF)")
    ax3.plot(t, flujo, color=COLOR_PROCESADA, lw=1.1)
    ax3.set_title("Flujo glotal estimado (fuente)")
    ax3.set_xlabel("Tiempo (ms)")

    ax4 = fig.add_subplot(rejilla[:, 1])
    f, h = freqz([1.0], vt, worN=1024, fs=fs)
    ax4.plot(f, 20 * np.log10(np.abs(h) + 1e-12), color=COLOR_ORIGINAL, lw=1.5)
    for i, (fr, _) in enumerate(lp.formantes(vt, fs), start=1):
        ax4.axvline(fr, ls="--", lw=0.8, color="0.5")
        ax4.text(fr, ax4.get_ylim()[1], f" F{i}", va="top", fontsize=8)
    ax4.set_xlim(0, fs / 2)
    ax4.set_title("Tracto vocal estimado 1/A(z)")
    ax4.set_xlabel("Frecuencia (Hz)")
    ax4.set_ylabel("dB")


def cepstrum(fig, x, fs, p: ParametrosAnalisis):
    """Cepstrum real, pico de pitch y liftering (envolvente vs. excitación)."""
    segmento, t0 = an.segmento_sonoro(x, fs, max(p.dur_ms, 40))
    w = an.crear_ventana(p.ventana, len(segmento))
    n_fft = max(1024, 2 ** int(np.ceil(np.log2(len(segmento)))))
    c = cp.cepstrum_real(segmento * w, n_fft)
    n_coef = cp.coeficientes_lifter(fs, 2.0)
    bajo, alto = cp.liftering(c, n_coef)

    ax1, ax2, ax3 = fig.subplots(3, 1)
    q = np.arange(n_fft // 2) / fs * 1000
    q_min, q_max = int(fs / 400), int(fs / 60)
    ax1.plot(q[1:], c[1 : n_fft // 2], color=COLOR_ORIGINAL, lw=0.8)
    pico = q_min + int(np.argmax(c[q_min:q_max]))
    ax1.plot(q[pico], c[pico], "o", color=COLOR_PROCESADA)
    ax1.annotate(f"Periodo {q[pico]:.2f} ms → F0 ≈ {fs / pico:.0f} Hz", (q[pico], c[pico]), xytext=(10, 5), textcoords="offset points", fontsize=8)
    ax1.axvspan(0, n_coef / fs * 1000, color=COLOR_SONORA, alpha=0.15, label="Lifter (envolvente)")
    ax1.set_xlim(0, min(25, q[-1]))
    ax1.set_ylim(-0.3, max(0.4, c[q_min:q_max].max() * 1.4))
    ax1.set_title(f"Cepstrum real (segmento en t = {t0:.2f} s)")
    ax1.set_xlabel("Quefrencia (ms)")
    ax1.legend(fontsize=8, loc="upper right")

    f = np.fft.rfftfreq(n_fft, 1 / fs)
    espectro = cp.espectro_desde_cepstrum(c)
    ax2.plot(f, espectro, color="0.6", lw=0.7, label="log|X(f)|")
    ax2.plot(f, cp.espectro_desde_cepstrum(bajo), color=COLOR_PROCESADA, lw=2, label=f"Envolvente cepstral ({n_coef} coef.)")
    sp = an.preenfasis(segmento, p.coef()) if p.preenfasis else segmento
    a, g = lp.lpc(sp * w, p.orden(fs))
    f_env, env = lp.envolvente_lpc(a, g, fs)
    ax2.plot(f_env, env - env.max() + cp.espectro_desde_cepstrum(bajo).max(), color=COLOR_ORIGINAL, lw=1.2, ls="--", label="Envolvente LPC")
    ax2.set_xlim(0, fs / 2)
    ax2.set_title("Liftering de baja quefrencia: envolvente del tracto vocal")
    ax2.set_ylabel("dB")
    ax2.legend(fontsize=8, loc="lower left")

    ax3.plot(f, cp.espectro_desde_cepstrum(alto), color=COLOR_SONORA, lw=0.7)
    ax3.set_xlim(0, min(fs / 2, 4000))
    ax3.set_title("Liftering de alta quefrencia: estructura armónica de la excitación")
    ax3.set_xlabel("Frecuencia (Hz)")
    ax3.set_ylabel("dB")


def comparacion(fig, x, y, fs, p: ParametrosAnalisis):
    """Original frente a procesada: formas de onda, espectrogramas y F0."""
    if y is None:
        ax = fig.subplots()
        ax.text(0.5, 0.5, "Todavía no hay señal procesada.\nAplica una modulación o un efecto.", ha="center", va="center")
        ax.set_axis_off()
        return
    ejes = fig.subplots(2, 2, sharey="row")
    f_max = min(fs / 2, 4000)
    for col, (senal, titulo, color) in enumerate(((x, "Original", COLOR_ORIGINAL), (y, "Procesada", COLOR_PROCESADA))):
        ejes[0, col].plot(_tiempo(senal, fs), senal, lw=0.6, color=color)
        ejes[0, col].set_title(f"{titulo} ({len(senal) / fs:.2f} s)")
        _dibujar_espectrograma(ejes[1, col], senal, fs, 40, 5, "hamming", f_max=f_max)
        f0, t = estimar_pitch(senal, fs, p.metodo_pitch)
        sonoras = f0 > 0
        mediana = np.median(f0[sonoras]) if np.any(sonoras) else 0
        ejes[1, col].plot(t[sonoras], f0[sonoras], ".", color="cyan", ms=3)
        ejes[1, col].set_title(f"F0 mediana: {mediana:.0f} Hz")
        ejes[1, col].set_xlabel("Tiempo (s)")
    ejes[0, 0].set_ylabel("Amplitud")


#: Gráficas disponibles: nombre visible -> función.
GRAFICAS = {
    "Señal y espectrograma": senal_y_espectrograma,
    "Energía y ZCR": energia_y_zcr,
    "Pitch": pitch,
    "LPC y formantes": lpc_y_formantes,
    "Filtrado inverso": filtrado_inverso,
    "Flujo glotal": flujo_glotal,
    "Cepstrum": cepstrum,
}

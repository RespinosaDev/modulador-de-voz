"""Genera las figuras de ``docs/img`` a partir del audio de ejemplo.

Uso:
    python scripts/generar_figuras.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modulador_voz import graficas  # noqa: E402
from modulador_voz.audio_io import guardar_audio  # noqa: E402
from modulador_voz.modificacion import cambiar_semitonos  # noqa: E402
from modulador_voz.sintetico import frase_sintetica  # noqa: E402

SALIDA = RAIZ / "docs" / "img"
ARCHIVOS = {
    "Señal y espectrograma": "espectrograma.png",
    "Energía y ZCR": "energia_zcr.png",
    "Pitch": "pitch.png",
    "LPC y formantes": "lpc_formantes.png",
    "Filtrado inverso": "filtrado_inverso.png",
    "Flujo glotal": "flujo_glotal.png",
    "Cepstrum": "cepstrum.png",
}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    x, fs = frase_sintetica(16000)
    guardar_audio(RAIZ / "ejemplos" / "frase_sintetica.wav", x, fs)
    p = graficas.ParametrosAnalisis()

    for nombre, archivo in ARCHIVOS.items():
        fig = Figure(figsize=(11, 6.5), dpi=110, layout="constrained")
        graficas.GRAFICAS[nombre](fig, x, fs, p)
        fig.suptitle(nombre, fontsize=12)
        fig.savefig(SALIDA / archivo)
        print("Guardada", SALIDA / archivo)

    y = cambiar_semitonos(x, fs, 5, "psola")
    guardar_audio(RAIZ / "ejemplos" / "frase_sintetica_+5_semitonos.wav", y, fs)
    fig = Figure(figsize=(11, 6.5), dpi=110, layout="constrained")
    graficas.comparacion(fig, x, y, fs, p)
    fig.suptitle("Original frente a +5 semitonos (PSOLA)", fontsize=12)
    fig.savefig(SALIDA / "comparacion_semitonos.png")
    print("Guardada", SALIDA / "comparacion_semitonos.png")


if __name__ == "__main__":
    main()

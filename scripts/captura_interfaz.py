"""Abre la interfaz, aplica una modulación y guarda capturas en ``docs/img``.

Uso (en Windows):
    python scripts/captura_interfaz.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

try:  # coordenadas de Tk y de la captura en píxeles físicos
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from PIL import ImageGrab  # noqa: E402

from modulador_voz.interfaz import AplicacionModulador  # noqa: E402

SALIDA = RAIZ / "docs" / "img"


def capturar(app, archivo):
    app.update()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(SALIDA / archivo)
    print("Guardada", SALIDA / archivo)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    app = AplicacionModulador()
    app.attributes("-topmost", True)

    def pasos():
        app.cargar_ejemplo()
        app.var_semitonos.set(5)
        app.var_velocidad.set(1.0)
        app.aplicar_modulacion()
        app.after(800, lambda: (capturar(app, "interfaz_comparacion.png"), paso2()))

    def paso2():
        app.var_fuente.set("original")
        app.graficar("Flujo glotal")
        app.after(800, lambda: (capturar(app, "interfaz_flujo_glotal.png"), app.destroy()))

    app.after(500, pasos)
    app.mainloop()


if __name__ == "__main__":
    main()

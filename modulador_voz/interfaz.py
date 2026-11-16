"""Interfaz gráfica (Tkinter + matplotlib) del modulador de voz."""

from __future__ import annotations

import ctypes
import functools
import sys
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import audio_io, graficas, lpc, modificacion, ruido  # noqa: E402
from .analisis import VENTANAS  # noqa: E402
from .pitch import METODOS_PITCH  # noqa: E402
from .sintetico import frase_sintetica  # noqa: E402

VELOCIDADES = (0.5, 1.0, 1.5, 2.0)
METODOS_TONO = {"PSOLA": "psola", "Phase vocoder": "vocoder"}
TIPOS_AUDIO = [
    ("Audio", "*.wav *.flac *.ogg *.mp3"),
    ("WAV", "*.wav"),
    ("FLAC", "*.flac"),
    ("OGG", "*.ogg"),
    ("MP3", "*.mp3"),
    ("Todos los archivos", "*.*"),
]


def _numero(valor: float) -> str:
    """Formatea un número con coma decimal: 1.5 -> «1,5»."""
    return f"{valor:g}".replace(".", ",")


def accion(mensaje: str | None = None):
    """Decorador para las acciones de la interfaz: cursor de espera y errores."""

    def decorador(funcion):
        @functools.wraps(funcion)
        def envoltura(self, *args, **kwargs):
            try:
                self.config(cursor="watch")
                if mensaje:
                    self._estado(f"{mensaje}…")
                self.update_idletasks()
                return funcion(self, *args, **kwargs)
            except Exception as error:  # se muestra al usuario en lugar de cerrar la app
                traceback.print_exc()
                messagebox.showerror("Error", str(error) or error.__class__.__name__, parent=self)
                self._estado("Error: " + (str(error) or error.__class__.__name__))
            finally:
                self.config(cursor="")

        return envoltura

    return decorador


class AplicacionModulador(tk.Tk):
    """Ventana principal: modulación a la izquierda, análisis a la derecha."""

    def __init__(self):
        super().__init__()
        self.title("Modulador de voz · Biometría de la Voz")
        escala = self.winfo_fpixels("1i") / 96  # 1,0 a 100 %, 1,5 a 150 %...
        ancho = min(int(self.winfo_screenwidth() * 0.9), int(1500 * escala))
        alto = min(int(self.winfo_screenheight() * 0.85), int(900 * escala))
        self.geometry(f"{ancho}x{alto}+20+20")
        self.minsize(int(1100 * escala), int(680 * escala))

        self.x = None        # señal original
        self.fs = None
        self.y = None        # señal procesada
        self.nombre = ""
        self.cadena: list[str] = []
        self.grafica_actual = "Señal y espectrograma"

        self._crear_variables()
        self._crear_estilos()
        self._construir()
        self._estado("Carga un archivo de audio o pulsa «Audio de ejemplo».")

    # ------------------------------------------------------------------ UI --

    def _crear_variables(self):
        self.var_semitonos = tk.IntVar(value=0)
        self.var_metodo_tono = tk.StringVar(value="PSOLA")
        self.var_velocidad = tk.DoubleVar(value=1.0)
        self.var_mantener_tono = tk.BooleanVar(value=True)
        self.var_f0_robot = tk.DoubleVar(value=110.0)
        self.var_snr = tk.DoubleVar(value=10.0)
        self.var_tipo_ruido = tk.StringVar(value="blanco")

        self.var_fuente = tk.StringVar(value="original")
        self.var_ventana = tk.StringVar(value="hamming")
        self.var_dur = tk.DoubleVar(value=30.0)
        self.var_salto = tk.DoubleVar(value=10.0)
        self.var_orden = tk.IntVar(value=0)
        self.var_preenfasis = tk.BooleanVar(value=True)
        self.var_metodo_pitch = tk.StringVar(value="autocorrelacion")

        self.var_info = tk.StringVar(value="Sin audio cargado")
        self.var_cadena = tk.StringVar(value="Procesada: sin cambios")
        self.var_estado = tk.StringVar()

    def _crear_estilos(self):
        estilo = ttk.Style(self)
        if "vista" in estilo.theme_names():
            estilo.theme_use("vista")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 13, "bold"))
        estilo.configure("Info.TLabel", foreground="#555555")
        estilo.configure("Accion.TButton", font=("Segoe UI", 9, "bold"))
        estilo.configure("Toolbutton", padding=(10, 4))

    def _construir(self):
        barra = ttk.Label(self, textvariable=self.var_estado, anchor="w", padding=(8, 3), relief="sunken")
        barra.pack(side="bottom", fill="x")

        izquierda = ttk.Frame(self, padding=(10, 8, 6, 8), width=330)
        izquierda.pack(side="left", fill="y")
        derecha = ttk.Frame(self, padding=(4, 8, 10, 8))
        derecha.pack(side="left", fill="both", expand=True)

        self._panel_archivo(izquierda)
        self._panel_reproduccion(izquierda)
        self._panel_modulacion(izquierda)
        self._panel_efectos(izquierda)
        self._panel_ruido(izquierda)
        ttk.Label(izquierda, textvariable=self.var_cadena, style="Info.TLabel", wraplength=300).pack(fill="x", pady=(4, 2))
        ttk.Button(izquierda, text="Restablecer (descartar cambios)", command=self.restablecer).pack(fill="x")

        self._panel_parametros(derecha)
        self._panel_graficas(derecha)

    def _seccion(self, padre, titulo):
        marco = ttk.LabelFrame(padre, text=titulo, padding=6)
        marco.pack(fill="x", pady=(0, 6))
        return marco

    def _panel_archivo(self, padre):
        ttk.Label(padre, text="Modulador de voz", style="Titulo.TLabel").pack(anchor="w", pady=(0, 6))
        f = self._seccion(padre, "Archivo")
        ttk.Button(f, text="Cargar audio…", command=self.cargar).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(f, text="Audio de ejemplo", command=self.cargar_ejemplo).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(f, text="Guardar / convertir formato…", command=self.guardar).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )
        ttk.Label(f, textvariable=self.var_info, style="Info.TLabel", wraplength=290).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=2
        )
        f.columnconfigure((0, 1), weight=1)

    def _panel_reproduccion(self, padre):
        f = self._seccion(padre, "Reproducción")
        ttk.Button(f, text="▶ Original", command=lambda: self.reproducir("original")).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(f, text="▶ Procesada", command=lambda: self.reproducir("procesada")).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(f, text="■ Parar", command=self.parar).grid(row=0, column=2, sticky="ew", padx=2)
        f.columnconfigure((0, 1, 2), weight=1)

    def _panel_modulacion(self, padre):
        f = self._seccion(padre, "Modulación (tono y velocidad)")
        ttk.Label(f, text="Semitonos").grid(row=0, column=0, sticky="w")
        tk.Scale(
            f, from_=-12, to=12, orient="horizontal", resolution=1, variable=self.var_semitonos, length=200, showvalue=True
        ).grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Label(f, text="Método").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.var_metodo_tono, values=list(METODOS_TONO), state="readonly", width=14).grid(
            row=1, column=1, columnspan=3, sticky="w", pady=2
        )

        ttk.Label(f, text="Velocidad").grid(row=2, column=0, sticky="w", pady=(6, 0))
        botones = ttk.Frame(f)
        botones.grid(row=2, column=1, columnspan=3, sticky="w", pady=(6, 0))
        for v in VELOCIDADES:
            ttk.Radiobutton(
                botones,
                text=f"x{_numero(v)}",
                value=v,
                variable=self.var_velocidad,
                style="Toolbutton",
                command=self.aplicar_modulacion,
            ).pack(side="left", padx=1)
        ttk.Checkbutton(f, text="Mantener el tono al cambiar la velocidad", variable=self.var_mantener_tono).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=2
        )
        ttk.Button(f, text="Aplicar modulación", style="Accion.TButton", command=self.aplicar_modulacion).grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0)
        )
        f.columnconfigure(1, weight=1)

    def _panel_efectos(self, padre):
        f = self._seccion(padre, "Efectos por síntesis LPC")
        ttk.Button(f, text="Robot", command=lambda: self.efecto_lpc("robot")).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(f, text="Susurro", command=lambda: self.efecto_lpc("susurro")).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(f, text="Vocoder LPC", command=lambda: self.efecto_lpc("original")).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Label(f, text="F0 del robot (Hz)").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Spinbox(f, from_=50, to=400, increment=10, textvariable=self.var_f0_robot, width=6).grid(
            row=1, column=2, sticky="w", pady=(4, 0)
        )
        f.columnconfigure((0, 1, 2), weight=1)

    def _panel_ruido(self, padre):
        f = self._seccion(padre, "Ruido")
        ttk.Label(f, text="SNR (dB)").grid(row=0, column=0, sticky="w")
        tk.Scale(f, from_=-5, to=30, orient="horizontal", resolution=1, variable=self.var_snr, length=160).grid(
            row=0, column=1, columnspan=2, sticky="ew"
        )
        ttk.Combobox(f, textvariable=self.var_tipo_ruido, values=list(ruido.TIPOS_RUIDO), state="readonly", width=7).grid(
            row=0, column=3, sticky="e"
        )
        ttk.Button(f, text="Añadir ruido", command=self.anadir_ruido).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Button(f, text="Sustracción espectral", command=self.sustraccion).grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=2, pady=2
        )
        f.columnconfigure(1, weight=1)

    def _panel_parametros(self, padre):
        f = ttk.LabelFrame(padre, text="Parámetros de análisis", padding=6)
        f.pack(fill="x")
        ttk.Label(f, text="Analizar:").pack(side="left")
        ttk.Radiobutton(f, text="Original", value="original", variable=self.var_fuente, command=self.redibujar).pack(side="left")
        ttk.Radiobutton(f, text="Procesada", value="procesada", variable=self.var_fuente, command=self.redibujar).pack(
            side="left", padx=(0, 12)
        )

        def campo(texto, widget):
            ttk.Label(f, text=texto).pack(side="left", padx=(8, 2))
            widget.pack(side="left")

        campo("Ventana", ttk.Combobox(f, textvariable=self.var_ventana, values=VENTANAS, state="readonly", width=10))
        campo("Trama (ms)", ttk.Spinbox(f, from_=5, to=100, increment=5, textvariable=self.var_dur, width=5))
        campo("Salto (ms)", ttk.Spinbox(f, from_=1, to=50, increment=1, textvariable=self.var_salto, width=5))
        campo("Orden LPC (0 = auto)", ttk.Spinbox(f, from_=0, to=60, increment=1, textvariable=self.var_orden, width=4))
        campo("Pitch", ttk.Combobox(f, textvariable=self.var_metodo_pitch, values=METODOS_PITCH, state="readonly", width=15))
        ttk.Checkbutton(f, text="Preénfasis", variable=self.var_preenfasis).pack(side="left", padx=(10, 0))
        ttk.Button(f, text="Actualizar", command=self.redibujar).pack(side="right")

    def _panel_graficas(self, padre):
        botones = ttk.Frame(padre)
        botones.pack(fill="x", pady=(6, 4))
        for nombre in list(graficas.GRAFICAS) + ["Comparar original / procesada"]:
            ttk.Button(botones, text=nombre, command=lambda n=nombre: self.graficar(n)).pack(
                side="left", fill="x", expand=True, padx=1
            )

        self.figura = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        self.lienzo = FigureCanvasTkAgg(self.figura, master=padre)
        herramientas = NavigationToolbar2Tk(self.lienzo, padre, pack_toolbar=False)
        herramientas.update()
        herramientas.pack(side="bottom", fill="x")
        widget = self.lienzo.get_tk_widget()
        widget.pack(fill="both", expand=True)
        # En pantallas con escalado (125 %, 150 %...) matplotlib cambia los ppp
        # de la figura al mostrarse y la agranda. Se vuelve a ajustar la figura
        # al espacio realmente disponible.
        def ajustar_figura(_evento=None):
            widget.configure(width=10, height=10)
            self.update_idletasks()
            self.lienzo.resize(SimpleNamespace(width=widget.winfo_width(), height=widget.winfo_height()))

        widget.bind("<Map>", lambda _evento: self.after_idle(ajustar_figura), add="+")
        self._mensaje_en_figura("Carga un audio para ver sus gráficas.")

    # ------------------------------------------------------------ utilidades --

    def _estado(self, texto: str):
        self.var_estado.set(texto)

    def _requiere_audio(self):
        if self.x is None:
            raise ValueError("Primero carga un archivo de audio (o el audio de ejemplo).")

    def _base(self):
        """Señal sobre la que se aplican los efectos: la procesada si existe."""
        return self.y if self.y is not None else self.x

    def _actualizar_cadena(self):
        self.var_cadena.set("Procesada: " + (" → ".join(self.cadena) if self.cadena else "sin cambios"))

    def _parametros(self) -> graficas.ParametrosAnalisis:
        return graficas.ParametrosAnalisis(
            dur_ms=float(self.var_dur.get()),
            salto_ms=float(self.var_salto.get()),
            ventana=self.var_ventana.get(),
            orden_lpc=int(self.var_orden.get()) or None,
            preenfasis=bool(self.var_preenfasis.get()),
            metodo_pitch=self.var_metodo_pitch.get(),
        )

    def _mensaje_en_figura(self, texto: str):
        self.figura.clear()
        ax = self.figura.subplots()
        ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=12, color="0.4")
        ax.set_axis_off()
        self.lienzo.draw_idle()

    def _nuevo_audio(self, x, fs, nombre):
        self.x, self.fs, self.nombre = x, fs, nombre
        self.y = None
        self.cadena = []
        self._actualizar_cadena()
        self.var_fuente.set("original")
        self.var_info.set(f"{nombre}\n{fs} Hz · {len(x) / fs:.2f} s · {len(x)} muestras")
        self._estado(f"Cargado «{nombre}».")
        self.graficar("Señal y espectrograma")

    def _nueva_procesada(self, y, descripcion: str, reiniciar_cadena=False):
        self.y = audio_io.normalizar(y, max(0.1, min(0.99, float(abs(self.x).max())))) if y.size else y
        self.cadena = [descripcion] if reiniciar_cadena else self.cadena + [descripcion]
        self._actualizar_cadena()
        self.var_fuente.set("procesada")
        self._estado(f"Aplicado: {descripcion}. Duración {len(self.y) / self.fs:.2f} s.")
        self.graficar("Comparar original / procesada")

    # --------------------------------------------------------------- acciones --

    @accion("Cargando audio")
    def cargar(self):
        ruta = filedialog.askopenfilename(parent=self, title="Abrir audio", filetypes=TIPOS_AUDIO)
        if not ruta:
            return
        x, fs = audio_io.cargar_audio(ruta)
        if x.size == 0:
            raise ValueError("El archivo no contiene muestras de audio.")
        self._nuevo_audio(x, fs, Path(ruta).name)

    @accion("Generando audio de ejemplo")
    def cargar_ejemplo(self):
        x, fs = frase_sintetica(16000)
        self._nuevo_audio(x, fs, "ejemplo_sintetico (a-s-i-u)")

    @accion("Guardando")
    def guardar(self):
        self._requiere_audio()
        senal = self._base()
        sufijo = "_procesado" if self.y is not None else ""
        ruta = filedialog.asksaveasfilename(
            parent=self,
            title="Guardar / convertir",
            defaultextension=".wav",
            initialfile=Path(self.nombre).stem + sufijo,
            filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac"), ("OGG Vorbis", "*.ogg"), ("MP3", "*.mp3")],
        )
        if not ruta:
            return
        destino = audio_io.guardar_audio(ruta, senal, self.fs)
        que = "señal procesada" if self.y is not None else "señal original"
        self._estado(f"Guardada la {que} en «{destino.name}» ({destino.suffix[1:].upper()}).")

    @accion()
    def reproducir(self, cual: str):
        self._requiere_audio()
        if cual == "procesada" and self.y is None:
            raise ValueError("Todavía no hay señal procesada. Aplica una modulación o un efecto.")
        senal = self.x if cual == "original" else self.y
        audio_io.reproducir(senal, self.fs)
        self._estado(f"Reproduciendo la señal {cual} ({len(senal) / self.fs:.2f} s).")

    @accion()
    def parar(self):
        audio_io.detener()
        self._estado("Reproducción detenida.")

    @accion("Aplicando modulación")
    def aplicar_modulacion(self):
        self._requiere_audio()
        semitonos = int(self.var_semitonos.get())
        velocidad = float(self.var_velocidad.get())
        mantener = bool(self.var_mantener_tono.get())
        nombre_metodo = self.var_metodo_tono.get()

        y = modificacion.cambiar_semitonos(self.x, self.fs, semitonos, METODOS_TONO[nombre_metodo])
        y = modificacion.cambiar_velocidad(y, self.fs, velocidad, mantener)

        partes = []
        if semitonos:
            partes.append(f"{semitonos:+d} semitonos ({nombre_metodo})")
        if velocidad != 1:
            partes.append(f"velocidad x{_numero(velocidad)}" + ("" if mantener else " (cambia el tono)"))
        self._nueva_procesada(y, ", ".join(partes) or "sin modulación", reiniciar_cadena=True)

    @accion("Sintetizando con LPC")
    def efecto_lpc(self, modo: str):
        self._requiere_audio()
        y = lpc.sintesis_lpc(self._base(), self.fs, modo, f0_robot=float(self.var_f0_robot.get()))
        nombres = {"robot": f"robot ({_numero(float(self.var_f0_robot.get()))} Hz)", "susurro": "susurro", "original": "vocoder LPC"}
        self._nueva_procesada(y, nombres[modo])

    @accion("Añadiendo ruido")
    def anadir_ruido(self):
        self._requiere_audio()
        snr = float(self.var_snr.get())
        y, _ = ruido.anadir_ruido(self._base(), snr, self.var_tipo_ruido.get())
        self._nueva_procesada(y, f"ruido {self.var_tipo_ruido.get()} SNR {_numero(snr)} dB")

    @accion("Aplicando sustracción espectral")
    def sustraccion(self):
        self._requiere_audio()
        base = self._base()
        y = ruido.sustraccion_espectral(base, self.fs)
        self._nueva_procesada(y, "sustracción espectral")

    @accion()
    def restablecer(self):
        self.y = None
        self.cadena = []
        self._actualizar_cadena()
        self.var_semitonos.set(0)
        self.var_velocidad.set(1.0)
        self.var_fuente.set("original")
        self._estado("Cambios descartados.")
        if self.x is not None:
            self.graficar(self.grafica_actual if self.grafica_actual != "Comparar original / procesada" else "Señal y espectrograma")

    @accion("Dibujando")
    def graficar(self, nombre: str):
        self._requiere_audio()
        self.grafica_actual = nombre
        parametros = self._parametros()
        self.figura.clear()
        if nombre == "Comparar original / procesada":
            graficas.comparacion(self.figura, self.x, self.y, self.fs, parametros)
        else:
            usar_procesada = self.var_fuente.get() == "procesada" and self.y is not None
            senal = self.y if usar_procesada else self.x
            graficas.GRAFICAS[nombre](self.figura, senal, self.fs, parametros)
            self.figura.suptitle(f"{nombre} · señal {'procesada' if usar_procesada else 'original'}", fontsize=11)
        self.lienzo.draw_idle()
        self._estado(f"Gráfica: {nombre}.")

    def redibujar(self):
        if self.x is not None:
            self.graficar(self.grafica_actual)


def activar_alta_resolucion():
    """En Windows, evita que el sistema escale la ventana como un mapa de bits (borrosa)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def main():
    activar_alta_resolucion()
    app = AplicacionModulador()
    app.mainloop()


if __name__ == "__main__":
    main()

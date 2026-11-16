"""Modulador y analizador de voz.

Paquete con las herramientas de análisis y modificación de la señal de voz
desarrolladas para la asignatura de Biometría de la Voz:

- ``analisis``: preénfasis, enventanado, energía, cruces por cero,
  clasificación silencio / sordo / sonoro y espectrograma.
- ``pitch``: estimación de la frecuencia fundamental (autocorrelación, AMDF
  y cepstrum).
- ``lpc``: predicción lineal (Levinson-Durbin), envolvente espectral,
  formantes, filtrado inverso, flujo glotal (IAIF) y síntesis LPC.
- ``cepstrum``: cepstrum real y liftering.
- ``modificacion``: cambio de tono en semitonos (PSOLA y phase vocoder) y
  cambio de velocidad.
- ``ruido``: adición de ruido con SNR controlada y sustracción espectral.
- ``audio_io``: carga, guardado, conversión de formato y reproducción.
"""

__version__ = "1.0.0"

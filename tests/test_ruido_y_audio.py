import numpy as np
import pytest

from modulador_voz import audio_io
from modulador_voz import ruido as rd

FS = 16000


@pytest.mark.parametrize("tipo", rd.TIPOS_RUIDO)
@pytest.mark.parametrize("snr", [0, 10, 20])
def test_anadir_ruido_con_snr(frase, tipo, snr):
    ruidosa, _ = rd.anadir_ruido(frase, snr, tipo, semilla=0)
    assert rd.snr_db(frase, ruidosa) == pytest.approx(snr, abs=0.01)


def test_sustraccion_espectral_mejora_snr(frase):
    ruidosa, _ = rd.anadir_ruido(frase, 5, semilla=0)
    filtrada = rd.sustraccion_espectral(ruidosa, FS)
    assert filtrada.shape == frase.shape
    assert rd.snr_db(frase, filtrada) > rd.snr_db(frase, ruidosa) + 3


@pytest.mark.parametrize("extension", [".wav", ".flac", ".ogg", ".mp3"])
def test_guardar_y_cargar(tmp_path, frase, extension):
    ruta = audio_io.guardar_audio(tmp_path / f"prueba{extension}", frase, FS)
    x, fs = audio_io.cargar_audio(ruta)
    assert fs == FS
    assert x.size == pytest.approx(frase.size, rel=0.08)  # MP3/OGG añaden relleno
    if extension in (".wav", ".flac"):
        np.testing.assert_allclose(x, frase, atol=1e-4)


def test_convertir_archivo(tmp_path, frase):
    origen = audio_io.guardar_audio(tmp_path / "origen.wav", frase, FS)
    destino = audio_io.convertir_archivo(origen, tmp_path / "destino.flac", fs=8000)
    x, fs = audio_io.cargar_audio(destino)
    assert fs == 8000
    assert x.size == pytest.approx(frase.size / 2, abs=2)


def test_formato_no_admitido(tmp_path, frase):
    with pytest.raises(ValueError):
        audio_io.guardar_audio(tmp_path / "x.aiff", frase, FS)

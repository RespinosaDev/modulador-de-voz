import numpy as np
import pytest

from modulador_voz.audio_io import remuestrear
from modulador_voz.pitch import METODOS_PITCH, estimar_pitch

FS = 16000


def mediana_f0(x, fs, metodo="autocorrelacion"):
    f0, _ = estimar_pitch(x, fs, metodo)
    return np.median(f0[f0 > 0])


@pytest.mark.parametrize("metodo", METODOS_PITCH)
def test_pitch_vocal_120hz(vocal_a, metodo):
    x, _ = vocal_a
    assert mediana_f0(x, FS, metodo) == pytest.approx(120, rel=0.03)


def test_pitch_a_44100hz(vocal_a):
    x, _ = vocal_a
    x44 = remuestrear(x, FS, 44100)
    assert mediana_f0(x44, 44100) == pytest.approx(120, rel=0.03)


def test_sin_pitch_en_silencio_y_fricativa(frase):
    f0, t = estimar_pitch(frase, FS)
    assert np.all(f0[t < 0.2] == 0)                    # silencio inicial
    assert np.all(f0[(t > 0.8) & (t < 0.88)] == 0)    # fricativa /s/

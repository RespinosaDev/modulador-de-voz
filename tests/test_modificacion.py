import numpy as np
import pytest

from modulador_voz import modificacion as md
from modulador_voz.pitch import estimar_pitch

FS = 16000


def mediana_f0(x):
    f0, _ = estimar_pitch(x, FS)
    return np.median(f0[f0 > 0])


def test_factor_semitonos():
    assert md.factor_semitonos(12) == pytest.approx(2.0)
    assert md.factor_semitonos(-12) == pytest.approx(0.5)
    assert md.factor_semitonos(7) == pytest.approx(1.4983, rel=1e-4)


@pytest.mark.parametrize("metodo", md.METODOS_TONO)
@pytest.mark.parametrize("semitonos", [-5, 4, 12])
def test_cambio_de_tono(vocal_a, metodo, semitonos):
    x, _ = vocal_a
    y = md.cambiar_semitonos(x, FS, semitonos, metodo)
    assert y.size == x.size  # la duración no cambia
    esperado = 120 * md.factor_semitonos(semitonos)
    assert mediana_f0(y) == pytest.approx(esperado, rel=0.05)


@pytest.mark.parametrize("velocidad", [0.5, 1.5, 2.0])
def test_velocidad_mantiene_tono(vocal_a, velocidad):
    x, _ = vocal_a
    y = md.cambiar_velocidad(x, FS, velocidad, mantener_tono=True)
    assert y.size == round(x.size / velocidad)
    assert mediana_f0(y) == pytest.approx(120, rel=0.05)


def test_velocidad_sin_mantener_tono(vocal_a):
    x, _ = vocal_a
    y = md.cambiar_velocidad(x, FS, 1.5, mantener_tono=False)
    assert y.size == round(x.size / 1.5)
    assert mediana_f0(y) == pytest.approx(180, rel=0.05)


def test_sin_cambios_devuelve_copia(vocal_a):
    x, _ = vocal_a
    y = md.cambiar_semitonos(x, FS, 0)
    np.testing.assert_array_equal(x, y)
    assert y is not x

import numpy as np
import pytest
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from modulador_voz import analisis as an
from modulador_voz import lpc as lp
from modulador_voz.sintetico import VOCALES

FS = 16000


def test_levinson_coincide_con_toeplitz():
    x = np.random.default_rng(1).standard_normal(400)
    orden = 10
    r = lp.autocorrelacion(x, orden)
    a, error, k = lp.levinson_durbin(r, orden)
    esperado = solve_toeplitz(r[:-1], -r[1:])
    np.testing.assert_allclose(a[1:], esperado, rtol=1e-8, atol=1e-10)
    assert error > 0
    assert np.all(np.abs(k) < 1)  # filtro estable


def test_recupera_filtro_ar():
    # Proceso AR(2) conocido: el LPC de orden 2 debe recuperar sus coeficientes.
    a_real = np.array([1.0, -1.3, 0.8])
    x = lfilter([1.0], a_real, np.random.default_rng(2).standard_normal(50000))
    a, _ = lp.lpc(x, 2)
    np.testing.assert_allclose(a, a_real, atol=0.02)


@pytest.mark.parametrize("vocal", ["a", "i", "u"])
def test_formantes_vocal_sintetica(vocal):
    from modulador_voz.sintetico import vocal_sintetica

    x, _ = vocal_sintetica(vocal, FS, 0.3, 120.0)
    segmento = x[2000:2000 + 480]
    a, _ = lp.lpc(an.preenfasis(segmento) * an.crear_ventana("hamming", 480), 12)
    estimados = [f for f, _ in lp.formantes(a, FS)]
    for f_real, _ in VOCALES[vocal][:2]:
        error_relativo = min(abs(f - f_real) / f_real for f in estimados)
        assert error_relativo < 0.10, (vocal, f_real, estimados)


def test_residuo_es_mas_plano(vocal_a):
    x, _ = vocal_a
    residuo = lp.residuo_lpc(x, FS)

    def planitud(s):  # media geométrica / media aritmética del espectro
        p = np.abs(np.fft.rfft(s[1000:5000] * np.hanning(4000))) ** 2 + 1e-12
        return np.exp(np.mean(np.log(p))) / np.mean(p)

    assert planitud(residuo) > 3 * planitud(x)


def test_iaif_separa_fuente_y_tracto(vocal_a):
    x, flujo_real = vocal_a
    ini, fin = 2400, 2400 + 1280  # 80 ms
    flujo, derivada, vt = lp.iaif(x[ini:fin], FS)

    # La derivada del flujo glotal estimada coincide con la real
    # (se compara la parte central para evitar los transitorios de los filtros).
    derivada_real = np.diff(flujo_real, prepend=0.0)[ini:fin]
    c = slice(200, -200)
    assert np.corrcoef(derivada[c], derivada_real[c])[0, 1] > 0.9

    # El tracto vocal estimado tiene los formantes de la /a/.
    estimados = [f for f, _ in lp.formantes(vt, FS)]
    for f_real, _ in VOCALES["a"]:
        assert min(abs(f - f_real) / f_real for f in estimados) < 0.05


@pytest.mark.parametrize("modo", ["robot", "susurro", "original"])
def test_sintesis_lpc(frase, modo):
    y = lp.sintesis_lpc(frase, FS, modo)
    assert y.shape == frase.shape
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(np.max(np.abs(y)), np.max(np.abs(frase)), rtol=1e-6)

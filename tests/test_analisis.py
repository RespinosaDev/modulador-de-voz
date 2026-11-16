import numpy as np

from modulador_voz import analisis as an

FS = 16000


def test_preenfasis():
    x = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(an.preenfasis(x, 0.9), [1.0, 0.1, 0.1])


def test_enventanar_dimensiones():
    x = np.random.default_rng(0).standard_normal(FS)  # 1 s
    tramas, tiempos = an.enventanar(x, FS, 30, 10)
    assert tramas.shape == (98, 480)
    assert tiempos[0] == 0.015


def test_zcr_senoide():
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * 500 * t)
    tramas, _ = an.enventanar(x, FS, 30, 10, "rectangular")
    # Una senoide de 500 Hz cruza por cero 1000 veces por segundo.
    np.testing.assert_allclose(an.tasa_cruces_cero(tramas) * FS, 1000, rtol=0.05)


def test_clasificacion_frase(frase):
    clases, t, *_ = an.clasificar_tramas(frase, FS)

    def clase_en(segundos):
        return clases[np.argmin(np.abs(t - segundos))]

    assert clase_en(0.10) == an.SILENCIO   # pausa inicial
    assert clase_en(0.45) == an.SONORO     # vocal /a/
    assert clase_en(0.85) == an.SORDO      # fricativa /s/
    assert clase_en(1.15) == an.SONORO     # vocal /i/


def test_espectrograma_dimensiones():
    x = np.zeros(FS)
    s, f, t = an.espectrograma(x, FS, 25, 5)
    assert s.shape == (len(f), len(t))
    assert f[-1] == FS / 2

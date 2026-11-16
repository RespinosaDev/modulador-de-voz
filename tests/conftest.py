import pytest

from modulador_voz.sintetico import frase_sintetica, vocal_sintetica

FS = 16000


@pytest.fixture(scope="session")
def vocal_a():
    """Vocal /a/ sintética de 0,5 s con F0 = 120 Hz. Devuelve (voz, flujo_glotal)."""
    return vocal_sintetica("a", FS, 0.5, 120.0)


@pytest.fixture(scope="session")
def frase():
    """Frase sintética «a - s - i - u» con pausas."""
    x, _ = frase_sintetica(FS)
    return x

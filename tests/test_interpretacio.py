from src.interpretacio import conclusio_logrank, interpretar_hr, interpretar_ph, text_mediana


def test_conclusio_logrank_significatiu():
    t = conclusio_logrank(1e-30, "tipus de contracte", millor="2 anys", pitjor="Mensual")
    assert "significativament diferents" in t and "< 0,001" in t
    assert "2 anys" in t and "Mensual" in t


def test_conclusio_logrank_no_significatiu():
    t = conclusio_logrank(0.30, "gènere")
    assert "no hi ha prou evidència" in t.lower() and "0,300" in t


def test_interpretar_hr():
    assert interpretar_hr(0.07, 1e-80) == "Redueix el risc un 93 % (significatiu)"
    assert interpretar_hr(2.24, 1e-20) == "Augmenta el risc un 124 % (significatiu)"
    assert interpretar_hr(1.03, 0.63).startswith("Sense efecte demostrat")


def test_interpretar_ph():
    assert "canvia amb el temps" in interpretar_ph(1e-8)
    assert "constant" in interpretar_ph(0.5)


def test_text_mediana():
    assert text_mediana("35") == "35"
    assert text_mediana("no assolida") == "> 72 (no s'assoleix)"

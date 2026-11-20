import numpy as np
import pandas as pd
import pytest

from src.perfils import (ESCENARIS, PERFILS, actius_perfil, afegir_tram, efecte_minim_rendible,
                         reduccio_maxima, resum_perfils, simular_campanya, taula_contracte_tram)
from src.preparacio import RUTA_DADES, carregar_dades
from src.risc_temporal import generar_scores


@pytest.fixture(scope="module")
def df():
    return carregar_dades(RUTA_DADES)


@pytest.fixture(scope="module")
def scores(df):
    return generar_scores(df)


def test_trams_cobreixen_tots_els_clients(df):
    d = afegir_tram(df)
    assert d["tram"].notna().all()
    assert list(d["tram"].cat.categories) == ["1–6", "7–12", "13–24", "25–48", "49–72"]


def test_mapa_contracte_tram(df):
    taxa, n = taula_contracte_tram(df)
    assert taxa.shape == (3, 5) and list(taxa.index) == ["Mensual", "1 any", "2 anys"]
    assert n.values.sum() == len(df)
    assert taxa.loc["Mensual", "1–6"] > 0.5                     # el grup de més risc
    assert (taxa.loc["Mensual"] > taxa.loc["2 anys"]).all()


def test_cada_perfil_te_mes_risc_que_la_referencia(df, scores):
    r = resum_perfils(df, scores)
    assert len(r) == len(PERFILS) == 4
    assert (r["taxa"] > r["taxa_ref"]).all()
    assert (r["actius"] > 0).all() and (r["n_ref"] > 0).all()
    assert r["risc_12m"].between(0, 1).all()
    assert np.allclose(r["baixes_esperades_12m"], r["risc_12m"] * r["actius"])


def test_actius_perfil_nomes_clients_actius(df, scores):
    a = actius_perfil(df, scores, "mensuals_consolidats")
    assert (a["event"] == 0).all()
    assert (a["Contract"] == "Month-to-month").all() and (a["tenure"] >= 12).all()


def test_simulador_campanya(df, scores):
    a = actius_perfil(df, scores, "mensuals_consolidats")
    red = reduccio_maxima(df)
    assert 0 < red < 1
    opt = simular_campanya(a, 0.2, 0.10, red, ESCENARIS["Optimista"])
    pru = simular_campanya(a, 0.2, 0.10, red, ESCENARIS["Prudent"])
    assert opt["accepten"] == round(len(a) * 0.2)
    assert opt["cost_descompte"] == pytest.approx(opt["accepten"] * opt["quota_mitjana"] * 0.10 * 12)
    assert pru["baixes_evitades"] == pytest.approx(opt["baixes_evitades"] / 2)
    assert opt["benefici"] > pru["benefici"]
    zero = simular_campanya(a, 0.2, 0.0, red, 1.0)
    assert zero["cost_descompte"] == 0 and zero["benefici"] > 0


def test_punt_equilibri_coherent(df, scores):
    a = actius_perfil(df, scores, "mensuals_consolidats")
    red = reduccio_maxima(df)
    minim = efecte_minim_rendible(a["risc_12m"].mean(), red, 0.10)
    res = simular_campanya(a, 0.2, 0.10, red, minim)
    assert res["benefici"] == pytest.approx(0, abs=1e-6)

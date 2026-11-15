from src.km import afegir_quartils_quota, ajustar_km, logrank, taula_medianes, taula_resum_km
from src.preparacio import carregar_dades


def test_km_per_contracte_ordre_esperat():
    f = ajustar_km(carregar_dades(), "Contract")
    s12 = {k: float(v.predict(12)) for k, v in f.items()}
    assert s12["Month-to-month"] < s12["One year"] < s12["Two year"]


def test_logrank_contracte_significatiu():
    assert logrank(carregar_dades(), "Contract")["p_valor"] < 1e-10


def test_mediana_no_assolida():
    t = taula_medianes(ajustar_km(carregar_dades(), "Contract")).set_index("grup")
    assert t.loc["Two year", "mediana"] == "no assolida"


def test_taula_resum_ordenada():
    t = taula_resum_km(afegir_quartils_quota(carregar_dades()))
    assert t["estadistic"].is_monotonic_decreasing and t.iloc[0]["variable"] == "Contract"


def test_km_global():
    f = ajustar_km(carregar_dades())
    assert list(f) == ["Global"]

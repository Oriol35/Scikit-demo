from src.cox import (ajustar_cox, c_index, comparar_parametrics, comprovar_ph,
                     preparar_estratificat, taula_hazard_ratios)
from src.preparacio import carregar_dades, codificar_cox, dividir_train_test


def test_cox_hr_contracte_protector():
    X = codificar_cox(carregar_dades())
    t = taula_hazard_ratios(ajustar_cox(X)).set_index("variable")
    assert t.loc["Contract_Two year", "HR"] < t.loc["Contract_One year", "HR"] < 1


def test_c_index_test_raonable():
    tr, te = dividir_train_test(carregar_dades())
    cph = ajustar_cox(codificar_cox(tr))
    assert 0.75 < c_index(cph, codificar_cox(te)) < 0.95


def test_ph_detecta_violacio():
    X = codificar_cox(carregar_dades())
    assert comprovar_ph(ajustar_cox(X), X)["viola"].any()


def test_estratificat_sense_dummies_contracte():
    cph = ajustar_cox(preparar_estratificat(carregar_dades()), strata=["Contract"])
    assert not any(v.startswith("Contract_") for v in cph.params_.index)


def test_parametrics():
    t = comparar_parametrics(carregar_dades())
    assert set(t["model"]) == {"Weibull", "Exponencial"} and t["AIC"].notna().all()

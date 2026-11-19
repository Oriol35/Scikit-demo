import numpy as np
import pytest

from src.classificacio import (MODELS, ajustar_model, avaluar, cami_arbre, campanya,
                               codificar_classificacio, codificar_perfil_classificacio,
                               corba_guanys, estructura_arbre, frase_metriques,
                               importancia_variables, interpretar_or, odds_ratios,
                               preparar_dades, probabilitats)
from src.preparacio import carregar_dades, dividir_train_test

PERFIL_RISC = {
    "tenure": 3, "MonthlyCharges": 85.0, "SeniorCitizen": 0, "Contract": "Month-to-month",
    "InternetService": "Fiber optic", "PaymentMethod": "Electronic check", "OnlineSecurity": "No",
    "TechSupport": "No", "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "PaperlessBilling": "Yes", "Partner": "No", "Dependents": "No", "gender": "Male",
}


@pytest.fixture(scope="module")
def dades():
    return preparar_dades(carregar_dades())


@pytest.fixture(scope="module")
def models_pesos(dades):
    return {nom: ajustar_model(nom, "Pesos de classe", dades["X_tr"], dades["y_tr"])
            for nom in MODELS}


def test_codificacio_sense_totalcharges_ni_valors_redundants():
    X = codificar_classificacio(carregar_dades())
    assert "TotalCharges" not in X.columns
    assert list(X.columns[:2]) == ["tenure", "MonthlyCharges"]   # numèriques primer (escalat)
    assert "Contract_Month-to-month" not in X.columns              # categoria de referència
    assert "PaymentMethod_Mailed check" not in X.columns           # mateixa referència que Cox
    assert not any("No internet service" in c or "No phone service" in c for c in X.columns)
    assert X.isna().sum().sum() == 0


def test_mateixa_particio_que_supervivencia(dades):
    _, te = dividir_train_test(carregar_dades())
    assert len(dades["X_te"]) == 1407
    assert list(dades["test"]["customerID"]) == list(te["customerID"])
    assert dades["y_te"].sum() == te["event"].sum()


def test_perfil_sense_internet_forca_serveis(dades):
    perfil = {**PERFIL_RISC, "InternetService": "No", "TechSupport": "Yes", "StreamingTV": "Yes",
              "PhoneService": "No", "MultipleLines": "Yes"}
    fila = codificar_perfil_classificacio(perfil, dades["columnes"])
    assert list(fila.columns) == dades["columnes"] and len(fila) == 1
    assert fila["TechSupport_Yes"].iloc[0] == 0 and fila["StreamingTV_Yes"].iloc[0] == 0
    assert fila["MultipleLines_Yes"].iloc[0] == 0


@pytest.mark.parametrize("nom", MODELS)
def test_tots_els_models_prediuen_be(dades, models_pesos, nom):
    p = probabilitats(models_pesos[nom], dades["X_te"])
    assert p.shape == (1407,) and (p >= 0).all() and (p <= 1).all()
    assert avaluar(dades["y_te"], p)["auc"] > 0.78


def test_sense_smote_no_es_carrega_imbalanced_learn(dades):
    import sys
    ajustar_model("Arbre de decisió", "Pesos de classe", dades["X_tr"], dades["y_tr"])
    if "imblearn" not in sys.modules:   # si cap test anterior l'ha importat
        assert "sklearn.cluster" not in sys.modules


def test_smote_equilibra_nomes_entrenament(dades):
    pytest.importorskip("imblearn")
    model = ajustar_model("Regressió logística", "SMOTE", dades["X_tr"], dades["y_tr"])
    Xt = model.named_steps["escalat"].transform(dades["X_tr"].to_numpy(dtype=float))
    _, y_res = model.named_steps["smote"].fit_resample(Xt, dades["y_tr"])
    assert y_res.mean() == pytest.approx(0.5)              # classes equilibrades a l'entrenament
    assert len(probabilitats(model, dades["X_te"])) == 1407  # la prova no es toca


def test_pesos_de_classe_milloren_el_recall(dades, models_pesos):
    sense = ajustar_model("Regressió logística", "Cap", dades["X_tr"], dades["y_tr"])
    r_sense = avaluar(dades["y_te"], probabilitats(sense, dades["X_te"]))["recall"]
    r_pesos = avaluar(dades["y_te"], probabilitats(models_pesos["Regressió logística"],
                                                   dades["X_te"]))["recall"]
    assert r_pesos > r_sense + 0.1


def test_matriu_confusio_consistent(dades, models_pesos):
    p = probabilitats(models_pesos["Gradient boosting"], dades["X_te"])
    m = avaluar(dades["y_te"], p, 0.5)
    assert m["vp"] + m["fp"] + m["fn"] + m["vn"] == 1407
    assert m["recall"] == pytest.approx(m["vp"] / (m["vp"] + m["fn"]))
    assert avaluar(dades["y_te"], p, 0.2)["recall"] >= m["recall"]   # baixar el llindar
    assert "van marxar" in frase_metriques(m)


def test_campanya_i_guanys(dades, models_pesos):
    y = dades["y_te"]
    p = probabilitats(models_pesos["Random Forest"], dades["X_te"])
    tot = campanya(y, p, 1.0)
    assert tot["pct_capturat"] == 1 and tot["lift"] == pytest.approx(1)
    deu = campanya(y, p, 0.1)
    assert deu["contactats"] == 141 and deu["lift"] > 2
    g = corba_guanys(y, p)
    assert g["pct_capturat"].iloc[0] == 0 and g["pct_capturat"].iloc[-1] == 1


def test_odds_ratios_coherents_amb_cox(dades, models_pesos):
    t = odds_ratios(models_pesos["Regressió logística"], dades["columnes"]).set_index("variable")
    assert t.loc["Contract_Two year", "OR"] < 1 < t.loc["InternetService_Fiber optic", "OR"]
    assert t.loc["tenure", "OR"] < 1                       # més antiguitat, menys risc
    with pytest.raises(TypeError):
        odds_ratios(models_pesos["Random Forest"], dades["columnes"])
    assert interpretar_or(0.25).startswith("Redueix") and interpretar_or(3).startswith("Multiplica")


def test_arbre_regles_i_cami(dades, models_pesos):
    arbre = models_pesos["Arbre de decisió"]
    est = estructura_arbre(arbre, dades["columnes"], dades["X_tr"], dades["y_tr"])
    assert est.loc[0, "n"] == 5625 and est["profunditat"].max() <= 4
    assert (est.loc[~est["fulla"], "pregunta"] != "").all()
    fila = codificar_perfil_classificacio(PERFIL_RISC, dades["columnes"])
    passos = cami_arbre(arbre, fila, est)
    assert passos[-1]["pregunta"] == "Resultat" and len(passos) >= 2
    assert all(p["resposta"] in ("Sí", "No") for p in passos[:-1])


def test_importancia_agrupada_per_variable(dades, models_pesos):
    imp = importancia_variables(models_pesos["Gradient boosting"], dades["X_te"], dades["y_te"],
                                repeticions=2)
    assert len(imp) == 18                                  # una fila per variable original
    assert set(imp["variable"].head(3)) & {"tenure", "Contract"}
    assert np.isfinite(imp["importancia"]).all()

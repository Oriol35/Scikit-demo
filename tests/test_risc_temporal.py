from src.preparacio import carregar_dades
from src.risc_temporal import generar_scores, hazard_mensual, mesos_critics


def test_hazard_mes1_es_el_mes_alt():
    hz = hazard_mensual(carregar_dades())
    assert hz.loc[hz["hazard"].idxmax(), "mes"] <= 3


def test_hazard_per_grup():
    hz = hazard_mensual(carregar_dades(), "Contract")
    assert set(hz["grup"]) == {"Month-to-month", "One year", "Two year"}


def test_mesos_critics_ignora_pocs_en_risc():
    hz = hazard_mensual(carregar_dades())
    crit = mesos_critics(hz)
    assert len(crit) == 5 and crit == sorted(crit)
    assert (hz.set_index("mes").loc[crit, "en_risc"] >= 50).all()


def test_scores_columnes_i_rangs():
    s = generar_scores(carregar_dades())
    assert list(s.columns) == ["customerID", "risc_cox", "S_6m", "S_12m",
                               "mesos_esperats_24m", "clv_esperat_24m", "segment_risc", "client_actiu"]
    assert len(s) == 7032 and s["S_12m"].between(0, 1).all()
    assert (s["S_6m"] >= s["S_12m"]).all()
    assert set(s["segment_risc"]) == {"Baix", "Mitjà", "Alt"}
    assert s["mesos_esperats_24m"].between(0, 24).all()


def test_supervivencia_condicional_no_es_congela_despres_del_mes_72():
    s = generar_scores(carregar_dades())
    df = carregar_dades()
    ant72 = s[df["duration"].values == 72]
    assert (ant72["S_12m"] < 1).all() and (ant72["mesos_esperats_24m"] < 24).all()
    ant70 = s[df["duration"].values == 70]
    assert (ant70["S_6m"] > ant70["S_12m"]).all()


def test_scores_marquen_clients_actius():
    df = carregar_dades()
    s = generar_scores(df)
    assert (s["client_actiu"].values == (df["event"].values == 0).astype(int)).all()


def test_corba_perfil_comenca_al_100_i_coincideix_amb_lifelines():
    import numpy as np
    from src.cox import ajustar_cox
    from src.preparacio import codificar_cox, codificar_perfil
    from src.risc_temporal import corba_perfil
    df = carregar_dades()
    X = codificar_cox(df)
    cph = ajustar_cox(X)
    cols = list(X.columns.drop(["duration", "event"]))
    fila = codificar_perfil(dict(Contract="Month-to-month", InternetService="Fiber optic",
                                 PaymentMethod="Electronic check", MonthlyCharges=85.0,
                                 OnlineSecurity="No", TechSupport="No", PaperlessBilling="Yes",
                                 Partner="No", Dependents="No", SeniorCitizen=0), cols)
    s = corba_perfil(cph, fila)
    assert s.loc[0] == 1.0 and list(s.index) == list(range(73))
    ref = cph.predict_survival_function(fila, times=np.arange(1, 73)).iloc[:, 0].to_numpy()
    assert np.allclose(s.loc[1:].to_numpy(), ref)

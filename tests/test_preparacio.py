from src.preparacio import carregar_dades, codificar_cox, codificar_perfil, dividir_train_test


def test_carregar_dades_neteja():
    df = carregar_dades()
    assert len(df) == 7032
    assert (df["duration"] > 0).all()
    assert df["event"].sum() == 1869
    assert df["TotalCharges"].dtype == float


def test_codificar_cox_sense_totalcharges_ni_colinealitat():
    X = codificar_cox(carregar_dades())
    assert "TotalCharges" not in X.columns
    assert "Contract_Month-to-month" not in X.columns
    assert "OnlineSecurity_No internet service" not in X.columns
    assert {"duration", "event", "Contract_Two year"} <= set(X.columns)


def test_codificar_perfil_forca_serveis_sense_internet():
    cols = codificar_cox(carregar_dades()).columns.drop(["duration", "event"])
    fila = codificar_perfil({"InternetService": "No", "TechSupport": "Yes", "Contract": "One year",
                             "PaymentMethod": "Mailed check", "OnlineSecurity": "Yes",
                             "PaperlessBilling": "No", "SeniorCitizen": 0, "Partner": "No",
                             "Dependents": "No", "MonthlyCharges": 20.0}, list(cols))
    assert list(fila.columns) == list(cols) and len(fila) == 1
    assert fila["TechSupport_Yes"].iloc[0] == 0 and fila["OnlineSecurity_Yes"].iloc[0] == 0


def test_dividir_train_test_estratificat():
    tr, te = dividir_train_test(carregar_dades())
    assert len(te) == 1407  # sklearn: ceil(7032 * 0.2)
    assert abs(tr["event"].mean() - te["event"].mean()) < 0.01

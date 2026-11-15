"""Preparació de les dades: neteja, definició de temps/esdeveniment, codificació i partició."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ARREL = Path(__file__).resolve().parents[1]
RUTA_DADES = ARREL / "data" / "Telco-Customer-Churn.csv"

COVARIABLES_COX = [
    "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents", "MonthlyCharges",
]

# Categoria de referència de cada variable categòrica (la que no té dummy).
REFERENCIES = {
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "PaymentMethod": "Mailed check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaperlessBilling": "No",
    "Partner": "No",
    "Dependents": "No",
}

# Serveis addicionals que només existeixen si el client té internet.
SERVEIS_INTERNET = ["OnlineSecurity", "TechSupport"]


def carregar_dades(ruta: Path = RUTA_DADES) -> pd.DataFrame:
    """Carrega i neteja el dataset. Afegeix `duration` (= tenure) i `event` (1 = baixa)."""
    df = pd.read_csv(ruta)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype(float)
    # Clients amb tenure 0: cap temps observat en risc i TotalCharges buit -> s'exclouen.
    df = df[df["tenure"] > 0].reset_index(drop=True)
    df["duration"] = df["tenure"].astype(int)
    df["event"] = (df["Churn"] == "Yes").astype(int)
    return df


def _normalitzar_serveis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SERVEIS_INTERNET:
        df[col] = df[col].replace("No internet service", "No")
    return df


def _dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica les covariables de Cox (sense duration/event)."""
    df = _normalitzar_serveis(df)
    parts = [df[["MonthlyCharges"]].astype(float), df[["SeniorCitizen"]].astype(int)]
    for var, ref in REFERENCIES.items():
        d = pd.get_dummies(df[var], prefix=var, prefix_sep="_", dtype=int)
        d = d.drop(columns=f"{var}_{ref}", errors="ignore")
        parts.append(d)
    return pd.concat(parts, axis=1)


def codificar_cox(df: pd.DataFrame) -> pd.DataFrame:
    """Matriu per a lifelines: duration, event i covariables codificades. Index = customerID."""
    X = _dummies(df)
    X.insert(0, "event", df["event"].values)
    X.insert(0, "duration", df["duration"].values)
    X.index = df["customerID"].values
    X.index.name = "customerID"
    return X


def codificar_perfil(perfil: dict, columnes: list[str]) -> pd.DataFrame:
    """Codifica un perfil de client simulat amb exactament les `columnes` del model."""
    perfil = dict(perfil)
    if perfil.get("InternetService") == "No":
        for col in SERVEIS_INTERNET:
            perfil[col] = "No"
    fila = pd.DataFrame([perfil])
    X = _dummies(fila)
    return X.reindex(columns=columnes, fill_value=0)


def dividir_train_test(df: pd.DataFrame, test_size: float = 0.2,
                       random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partició train/test estratificada per `event` (reproduïble amb `random_state`)."""
    tr, te = train_test_split(df, test_size=test_size, random_state=random_state,
                              stratify=df["event"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)

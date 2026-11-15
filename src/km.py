"""Estimador de Kaplan-Meier i test log-rank."""
import math

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

VARIABLES_KM = [
    "Contract", "InternetService", "PaymentMethod", "TechSupport",
    "OnlineSecurity", "SeniorCitizen", "Partner", "QuotaMensual_Q",
]

NOMS_VARIABLES = {
    "Contract": "Tipus de contracte",
    "InternetService": "Servei d'internet",
    "PaymentMethod": "Mètode de pagament",
    "TechSupport": "Suport tècnic",
    "OnlineSecurity": "Seguretat en línia",
    "SeniorCitizen": "Client sènior",
    "Partner": "Té parella",
    "QuotaMensual_Q": "Quota mensual (quartils)",
}

ETIQUETES_QUARTILS = ["Q1 (baixa)", "Q2", "Q3", "Q4 (alta)"]


def afegir_quartils_quota(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["QuotaMensual_Q"] = pd.qcut(df["MonthlyCharges"], 4, labels=ETIQUETES_QUARTILS).astype(str)
    return df


def ajustar_km(df: pd.DataFrame, grup: str | None = None) -> dict[str, KaplanMeierFitter]:
    """Un KaplanMeierFitter per nivell de `grup` (o un de sol, 'Global')."""
    if grup is None:
        kmf = KaplanMeierFitter(label="Global").fit(df["duration"], df["event"])
        return {"Global": kmf}
    fitters = {}
    for nivell, sub in sorted(df.groupby(grup), key=lambda kv: str(kv[0])):
        nom = str(nivell)
        fitters[nom] = KaplanMeierFitter(label=nom).fit(sub["duration"], sub["event"])
    return fitters


def logrank(df: pd.DataFrame, grup: str) -> dict:
    r = multivariate_logrank_test(df["duration"], df[grup].astype(str), df["event"])
    return {"estadistic": float(r.test_statistic), "p_valor": float(r.p_value)}


def taula_resum_km(df: pd.DataFrame, variables: list[str] = VARIABLES_KM) -> pd.DataFrame:
    files = [{"variable": v, **logrank(df, v)} for v in variables]
    return (pd.DataFrame(files).sort_values("estadistic", ascending=False)
            .reset_index(drop=True))


def taula_medianes(fitters: dict) -> pd.DataFrame:
    files = []
    for nom, kmf in fitters.items():
        med = kmf.median_survival_time_
        files.append({
            "grup": nom,
            "n": int(kmf.event_table["at_risk"].iloc[0]),
            "mediana": "no assolida" if math.isinf(med) else str(int(med)),
            "S_6m": round(float(kmf.predict(6)), 3),
            "S_12m": round(float(kmf.predict(12)), 3),
            "S_24m": round(float(kmf.predict(24)), 3),
        })
    return pd.DataFrame(files)

"""Model de riscos proporcionals de Cox i models paramètrics."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, ExponentialFitter, WeibullFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from src.preparacio import codificar_cox

NOMS_COVARIABLES = {
    "MonthlyCharges": "Quota mensual (+1 €)",
    "SeniorCitizen": "Client sènior",
    "Contract_One year": "Contracte 1 any (vs mensual)",
    "Contract_Two year": "Contracte 2 anys (vs mensual)",
    "InternetService_Fiber optic": "Fibra òptica (vs DSL)",
    "InternetService_No": "Sense internet (vs DSL)",
    "PaymentMethod_Bank transfer (automatic)": "Transferència automàtica (vs xec postal)",
    "PaymentMethod_Credit card (automatic)": "Targeta automàtica (vs xec postal)",
    "PaymentMethod_Electronic check": "Xec electrònic (vs xec postal)",
    "OnlineSecurity_Yes": "Seguretat en línia",
    "TechSupport_Yes": "Suport tècnic",
    "PaperlessBilling_Yes": "Factura electrònica",
    "Partner_Yes": "Té parella",
    "Dependents_Yes": "Té persones a càrrec",
}


def preparar_estratificat(df: pd.DataFrame) -> pd.DataFrame:
    """Matriu codificada amb `Contract` en brut (per estratificar) en lloc de les seves dummies."""
    X = codificar_cox(df)
    X = X.drop(columns=[c for c in X.columns if c.startswith("Contract_")])
    X["Contract"] = df["Contract"].values
    return X


def ajustar_cox(df_cod: pd.DataFrame, strata: list[str] | None = None,
                penalizer: float = 0.01) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_cod, duration_col="duration", event_col="event", strata=strata)
    return cph


def taula_hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary
    t = pd.DataFrame({
        "variable": s.index,
        "HR": s["exp(coef)"].values,
        "IC_inf": s["exp(coef) lower 95%"].values,
        "IC_sup": s["exp(coef) upper 95%"].values,
        "p_valor": s["p"].values,
    })
    return t.sort_values("HR").reset_index(drop=True)


def comprovar_ph(cph: CoxPHFitter, df_cod: pd.DataFrame) -> pd.DataFrame:
    """Test de riscos proporcionals basat en residus de Schoenfeld."""
    r = proportional_hazard_test(cph, df_cod, time_transform="rank")
    s = r.summary
    t = pd.DataFrame({"variable": s.index, "p_valor": s["p"].values})
    t["viola"] = t["p_valor"] < 0.05
    return t.sort_values("p_valor").reset_index(drop=True)


def c_index(cph: CoxPHFitter, df_cod: pd.DataFrame) -> float:
    risc = cph.predict_partial_hazard(df_cod)
    return float(concordance_index(df_cod["duration"], -np.asarray(risc), df_cod["event"]))


def comparar_parametrics(df: pd.DataFrame) -> pd.DataFrame:
    files = []
    for nom, fitter in [("Weibull", WeibullFitter()), ("Exponencial", ExponentialFitter())]:
        fitter.fit(df["duration"], df["event"])
        files.append({"model": nom, "AIC": float(fitter.AIC_)})
    return pd.DataFrame(files).sort_values("AIC").reset_index(drop=True)

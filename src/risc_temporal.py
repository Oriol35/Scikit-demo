"""Risc temporal (hazard mensual, mesos crítics), CLV esperat i fitxer de scores."""
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.cox import ajustar_cox
from src.preparacio import ARREL, codificar_cox

RUTA_SCORES = ARREL / "outputs" / "scores_supervivencia.csv"
MAX_MES = 72


def _hazard(df: pd.DataFrame) -> pd.DataFrame:
    mesos = np.arange(1, MAX_MES + 1)
    dur = df["duration"].to_numpy()
    ev = df["event"].to_numpy()
    en_risc = np.array([(dur >= m).sum() for m in mesos])
    baixes = np.array([((dur == m) & (ev == 1)).sum() for m in mesos])
    hazard = np.divide(baixes, en_risc, out=np.zeros(len(mesos)), where=en_risc > 0)
    return pd.DataFrame({"mes": mesos, "en_risc": en_risc, "baixes": baixes, "hazard": hazard})


def hazard_mensual(df: pd.DataFrame, grup: str | None = None) -> pd.DataFrame:
    """Taxa d'abandonament discreta per mes d'antiguitat: baixes / clients en risc."""
    if grup is None:
        return _hazard(df)
    parts = [_hazard(sub).assign(grup=str(nivell)) for nivell, sub in df.groupby(grup)]
    return pd.concat(parts, ignore_index=True)


def mesos_critics(hz: pd.DataFrame, n: int = 5, min_en_risc: int = 50) -> list[int]:
    """Els `n` mesos amb més risc, ignorant mesos amb pocs clients en risc (soroll)."""
    valids = hz[hz["en_risc"] >= min_en_risc]
    return sorted(int(m) for m in valids.nlargest(n, "hazard")["mes"])


def risc_base_acumulat(cph: CoxPHFitter, temps: np.ndarray, finestra: int = 12) -> np.ndarray:
    """H0(t) del model de Cox, extrapolat linealment més enllà de l'últim mes observat.

    Les dades acaben a 72 mesos; sense extrapolació H0 quedaria pla i S(t) deixaria de baixar
    (clients amb antiguitat alta tindrien risc 0). Es prolonga amb el pendent mitjà dels
    últims `finestra` mesos observats.
    """
    h0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t_max = float(h0.index.max())
    valors = h0.reindex(np.union1d(h0.index, np.arange(0, t_max + 1))).ffill().fillna(0.0)
    pendent = (valors.loc[t_max] - valors.loc[t_max - finestra]) / finestra
    temps = np.asarray(temps, dtype=float)
    dins = np.interp(np.minimum(temps, t_max), valors.index.to_numpy(dtype=float),
                     valors.to_numpy(), left=0.0)
    # Esglaonat: valor de H0 a l'últim temps <= t (np.interp amb enters coincideix als mesos).
    return np.where(temps > t_max, valors.loc[t_max] + pendent * (temps - t_max), dins)


def corba_perfil(cph: CoxPHFitter, fila: pd.DataFrame, horitzo: int = MAX_MES) -> pd.Series:
    """S(t) d'un client nou (antiguitat 0) per t = 0..horitzo, amb S(0) = 1.

    lifelines avalua t = 0 amb el primer valor del risc base (mes 1), i la corba començaria per
    sota del 100 %. Aquí H0(0) = 0 per definició.
    """
    ph = float(np.asarray(cph.predict_partial_hazard(fila), dtype=float)[0])
    t = np.arange(0, horitzo + 1)
    return pd.Series(np.exp(-risc_base_acumulat(cph, t) * ph), index=t)


def supervivencia_condicional(cph: CoxPHFitter, df_cod: pd.DataFrame,
                              horitzo: int = 24) -> pd.DataFrame:
    """P(continuar sent client d'aquí a t mesos | antiguitat actual), t = 0..horitzo.

    S(t | c, x) = exp(-[H0(c + t) - H0(c)] · exp(x·β)), amb H0 extrapolat més enllà de 72 mesos.
    """
    X = df_cod.drop(columns=["duration", "event"])
    ph = np.asarray(cph.predict_partial_hazard(X), dtype=float)[:, None]
    c = df_cod["duration"].to_numpy(dtype=float)[:, None]
    t = np.arange(0, horitzo + 1, dtype=float)[None, :]
    dH = risc_base_acumulat(cph, c + t) - risc_base_acumulat(cph, c)
    s = pd.DataFrame(np.exp(-dH * ph), index=df_cod.index, columns=list(range(0, horitzo + 1)))
    return s.clip(0, 1)


def clv_esperat(s_cond: pd.DataFrame, quota: pd.Series) -> pd.DataFrame:
    """Mesos esperats com a client en l'horitzó (Σ S(t), t=1..H) i ingrés esperat."""
    mesos = s_cond.loc[:, s_cond.columns[1:]].sum(axis=1)
    return pd.DataFrame({"mesos_esperats_24m": mesos,
                         "clv_esperat_24m": mesos * np.asarray(quota, dtype=float)},
                        index=s_cond.index)


def generar_scores(df: pd.DataFrame, horitzo: int = 24) -> pd.DataFrame:
    X = codificar_cox(df)
    cph = ajustar_cox(X)
    s = supervivencia_condicional(cph, X, horitzo)
    clv = clv_esperat(s, df["MonthlyCharges"])
    risc = np.asarray(cph.predict_partial_hazard(X.drop(columns=["duration", "event"])))
    scores = pd.DataFrame({
        "customerID": df["customerID"].values,
        "risc_cox": risc,
        "S_6m": s[6].values,
        "S_12m": s[12].values,
        "mesos_esperats_24m": clv["mesos_esperats_24m"].values,
        "clv_esperat_24m": clv["clv_esperat_24m"].values,
    })
    scores["segment_risc"] = pd.qcut(scores["risc_cox"], 3,
                                     labels=["Baix", "Mitjà", "Alt"]).astype(str)
    # S_6m, S_12m i el CLV només tenen sentit per als clients actius (censurats); per als que
    # ja han marxat són hipotètics (com si continuessin). Aquesta columna permet filtrar-los.
    scores["client_actiu"] = (df["event"].values == 0).astype(int)
    return scores


def desar_scores(scores: pd.DataFrame, ruta: Path = RUTA_SCORES) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    scores.round(4).to_csv(ruta, index=False)
    return ruta

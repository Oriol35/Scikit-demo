"""Perfils de risc i campanyes de retenció: de la predicció a l'acció.

Els clients s'agrupen en perfils fàcils d'explicar al negoci (contracte, antiguitat, internet i
mètode de pagament). Per a cada perfil es compara la taxa de baixa amb un grup de referència
semblant i s'hi associa una campanya concreta.

- La **taxa de baixa** d'un perfil és descriptiva: el % de clients del perfil que han marxat.
- El **risc dels clients actius** és la probabilitat de marxar en els propers 12 mesos segons el
  model de Cox (1 − S_12m de `generar_scores`), perquè té un horitzó temporal clar.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRAMS = [0, 6, 12, 24, 48, 72]
ETIQUETES_TRAMS = ["1–6", "7–12", "13–24", "25–48", "49–72"]
NOMS_CONTRACTE = {"Month-to-month": "Mensual", "One year": "1 any", "Two year": "2 anys"}
MESOS_CAMPANYA = 12   # horitzó del simulador: un any de quota


def afegir_tram(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix la columna `tram` (tram d'antiguitat en mesos)."""
    df = df.copy()
    df["tram"] = pd.cut(df["tenure"], TRAMS, labels=ETIQUETES_TRAMS)
    return df


def taula_contracte_tram(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Taxa de baixa i nombre de clients per tipus de contracte × tram d'antiguitat."""
    d = afegir_tram(df)
    taxa = d.pivot_table(index="Contract", columns="tram", values="event", aggfunc="mean",
                         observed=False)
    n = d.pivot_table(index="Contract", columns="tram", values="event", aggfunc="count",
                      observed=False)
    ordre = list(NOMS_CONTRACTE)
    return taxa.loc[ordre].rename(index=NOMS_CONTRACTE), n.loc[ordre].rename(index=NOMS_CONTRACTE)


# ---------- Perfils ----------
@dataclass(frozen=True)
class Perfil:
    clau: str
    nom: str
    qui: str                                   # descripció del perfil
    condicio: Callable[[pd.DataFrame], pd.Series]
    nom_referencia: str
    referencia: Callable[[pd.DataFrame], pd.Series]
    campanya: str
    accio: str                                 # què fer exactament


def _mensual(d: pd.DataFrame) -> pd.Series:
    return d["Contract"] == "Month-to-month"


PERFILS = [
    Perfil(
        clau="nous_mensuals", nom="Clients nous amb contracte mensual",
        qui="Contracte mensual i 6 mesos o menys d'antiguitat.",
        condicio=lambda d: _mensual(d) & (d["tenure"] <= 6),
        nom_referencia="mensuals amb més de 6 mesos",
        referencia=lambda d: _mensual(d) & (d["tenure"] > 6),
        campanya="Programa de benvinguda",
        accio="Trucada de seguiment el primer mes, revisió de la instal·lació i un petit avantatge "
              "durant els 3 primers mesos, que són els de més risc (pestanya Risc temporal)."),
    Perfil(
        clau="mensuals_consolidats", nom="Clients mensuals consolidats",
        qui="Contracte mensual i 12 mesos o més d'antiguitat.",
        condicio=lambda d: _mensual(d) & (d["tenure"] >= 12),
        nom_referencia="contracte d'1 any amb 12 mesos o més",
        referencia=lambda d: (d["Contract"] == "One year") & (d["tenure"] >= 12),
        campanya="Pas a contracte anual",
        accio="Oferir un descompte a canvi de passar a contracte d'1 any. L'efecte s'estima al "
              "simulador d'aquesta pestanya."),
    Perfil(
        clau="fibra_sense_suport", nom="Fibra sense suport ni seguretat",
        qui="Contracte mensual, fibra òptica i sense suport tècnic ni seguretat en línia.",
        condicio=lambda d: _mensual(d) & (d["InternetService"] == "Fiber optic")
        & (d["TechSupport"] == "No") & (d["OnlineSecurity"] == "No"),
        nom_referencia="mensuals amb fibra, suport tècnic i seguretat",
        referencia=lambda d: _mensual(d) & (d["InternetService"] == "Fiber optic")
        & (d["TechSupport"] == "Yes") & (d["OnlineSecurity"] == "Yes"),
        campanya="Suport i seguretat inclosos",
        accio="Activar gratis el suport tècnic i la seguretat en línia durant 3 mesos: són els "
              "serveis que més retenen (odds ratio ≈ 0,7 i hazard ratio ≈ 0,6–0,7)."),
    Perfil(
        clau="xec_electronic", nom="Pagament amb xec electrònic",
        qui="Contracte mensual i pagament amb xec electrònic.",
        condicio=lambda d: _mensual(d) & (d["PaymentMethod"] == "Electronic check"),
        nom_referencia="mensuals amb pagament automàtic (transferència o targeta)",
        referencia=lambda d: _mensual(d) & d["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]),
        campanya="Domiciliació amb incentiu",
        accio="Petit descompte o bonificació per passar a pagament automàtic: els clients que "
              "domicilien marxen força menys (hazard ratio ≈ 0,52 al model de Cox)."),
]
PERFILS_PER_CLAU = {p.clau: p for p in PERFILS}


def unir_risc(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Afegeix `risc_12m` (probabilitat de baixa en 12 mesos, Cox) a cada client."""
    return df.merge(scores[["customerID", "S_12m"]], on="customerID", how="left").assign(
        risc_12m=lambda d: 1 - d["S_12m"])


def resum_perfils(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Una fila per perfil: taxa de baixa vs referència i situació dels clients actius."""
    d = unir_risc(df, scores)
    files = []
    for p in PERFILS:
        dins, ref = p.condicio(d), p.referencia(d)
        actius = d[dins & (d["event"] == 0)]
        files.append({
            "clau": p.clau, "perfil": p.nom, "campanya": p.campanya,
            "n": int(dins.sum()), "taxa": float(d.loc[dins, "event"].mean()),
            "referencia": p.nom_referencia, "n_ref": int(ref.sum()),
            "taxa_ref": float(d.loc[ref, "event"].mean()),
            "actius": len(actius),
            "risc_12m": float(actius["risc_12m"].mean()),
            "baixes_esperades_12m": float(actius["risc_12m"].sum()),
            "quota_mitjana": float(actius["MonthlyCharges"].mean()),
            "ingressos_risc_mes": float((actius["risc_12m"] * actius["MonthlyCharges"]).sum()),
        })
    t = pd.DataFrame(files)
    t["diferencia_pp"] = (t["taxa"] - t["taxa_ref"]) * 100
    return t


def actius_perfil(df: pd.DataFrame, scores: pd.DataFrame, clau: str) -> pd.DataFrame:
    d = unir_risc(df, scores)
    return d[PERFILS_PER_CLAU[clau].condicio(d) & (d["event"] == 0)]


# ---------- Simulador de la campanya de contracte anual ----------
ESCENARIS = {"Optimista": 1.0, "Prudent": 0.5}


def reduccio_maxima(df: pd.DataFrame, clau: str = "mensuals_consolidats") -> float:
    """Reducció relativa del risc si el client es comportés com els de la referència.

    Per als mensuals consolidats: 1 − taxa(1 any, ≥ 12 mesos) / taxa(mensual, ≥ 12 mesos).
    És un màxim: part de la diferència es deu a qui tria el contracte anual (autoselecció),
    no al contracte en si.
    """
    p = PERFILS_PER_CLAU[clau]
    return float(1 - df.loc[p.referencia(df), "event"].mean() / df.loc[p.condicio(df), "event"].mean())


def simular_campanya(actius: pd.DataFrame, pct_accepten: float, descompte: float,
                     reduccio: float, efecte: float, mesos: int = MESOS_CAMPANYA) -> dict:
    """Escenari d'una campanya de descompte per passar a contracte anual.

    - `pct_accepten`: fracció de clients del perfil que accepten l'oferta (0–1).
    - `descompte`: descompte sobre la quota (0–1) durant `mesos`; el cobren TOTS els que accepten,
      també els que no haurien marxat (per això l'uplift és clau).
    - `reduccio`: reducció màxima del risc (vegeu `reduccio_maxima`).
    - `efecte`: fracció de la reducció màxima que es creu real (1 = optimista, 0,5 = prudent).
    Suposa que els que accepten tenen el risc i la quota mitjans del perfil.
    """
    n = len(actius)
    accepten = int(round(n * pct_accepten))
    risc = float(actius["risc_12m"].mean())
    quota = float(actius["MonthlyCharges"].mean())
    baixes_sense = accepten * risc
    evitades = baixes_sense * reduccio * efecte
    salvats = evitades * quota * mesos
    cost = accepten * quota * descompte * mesos
    return {"clients": n, "accepten": accepten, "risc_mitja": risc, "quota_mitjana": quota,
            "baixes_sense_campanya": baixes_sense, "baixes_evitades": evitades,
            "ingressos_salvats": salvats, "cost_descompte": cost, "benefici": salvats - cost}


def efecte_minim_rendible(risc: float, reduccio: float, descompte: float) -> float:
    """Fracció mínima de l'efecte màxim perquè la campanya no perdi diners.

    Benefici ≥ 0 ⇔ risc · reducció · efecte ≥ descompte ⇔ efecte ≥ descompte / (risc · reducció).
    """
    if risc * reduccio <= 0:
        return np.inf
    return descompte / (risc * reduccio)

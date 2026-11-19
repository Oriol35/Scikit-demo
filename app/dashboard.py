"""Dashboard del cas pràctic: anàlisi de supervivència i retenció (Telco Customer Churn).

Execució:  .venv/Scripts/python.exe -m streamlit run app/dashboard.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ARREL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARREL))
# Streamlit Cloud torna a executar aquest fitxer quan s'actualitza el repositori, però pot
# conservar a memòria versions antigues dels mòduls de src/. Els descartem perquè s'importi
# sempre el codi actual (si no, apareixen errors com "cannot import name ...").
for _mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_mod]

from src.cox import (NOMS_COVARIABLES, ajustar_cox, c_index, comprovar_ph,  # noqa: E402
                     preparar_estratificat, taula_hazard_ratios)
from src.grafics import (fig_corbes_client, fig_forest, fig_hazard, fig_km,  # noqa: E402
                         fig_risc_valor, traduir)
from src.interpretacio import (conclusio_logrank, frase_grups, interpretar_hr,  # noqa: E402
                               interpretar_ph, text_mediana)
from src.km import (NOMS_VARIABLES, VARIABLES_KM, afegir_quartils_quota,  # noqa: E402
                    ajustar_km, logrank, taula_medianes)
from src.preparacio import (RUTA_DADES, carregar_dades, codificar_cox,  # noqa: E402
                            codificar_perfil, dividir_train_test)
from src.risc_temporal import generar_scores, hazard_mensual, mesos_critics  # noqa: E402
from src.classificacio import (AJUDA_METRIQUES, DESCRIPCIONS_MODELS,  # noqa: E402
                               DESCRIPCIONS_TRACTAMENTS, MODELS, TRACTAMENTS, ajustar_tots,
                               avaluar, cami_arbre, campanya, codificar_perfil_classificacio,
                               corba_guanys, corba_roc, estructura_arbre, frase_metriques,
                               importancia_variables, interpretar_or, nodes_cami, odds_ratios,
                               preparar_dades, probabilitats, smote_disponible,
                               taula_comparativa)
from src.grafics_classificacio import (fig_arbre, fig_confusio, fig_guanys,  # noqa: E402
                                       fig_importancia, fig_odds_ratios,
                                       fig_probabilitats_perfil, fig_roc)

DIR_OUTPUTS = Path(os.environ.get("DIR_OUTPUTS", ARREL / "outputs"))
GRAFIC = dict(theme=None, config={"displaylogo": False})

st.set_page_config(page_title="Supervivència i retenció · Telco", layout="wide")


# ---------- Dades i models (es calculen un cop i es reutilitzen) ----------
@st.cache_resource(show_spinner="Ajustant els models…")
def models():
    df = afegir_quartils_quota(carregar_dades())
    X = codificar_cox(df)
    tr, te = dividir_train_test(df)
    X_tr, X_te = codificar_cox(tr), codificar_cox(te)
    cph_tr = ajustar_cox(X_tr)
    cph_e = ajustar_cox(preparar_estratificat(tr), strata=["Contract"])
    n_bruts = len(pd.read_csv(RUTA_DADES))
    quota = df.groupby(df["InternetService"] == "No")["MonthlyCharges"]
    return {
        "df": df,
        "n_exclosos": n_bruts - len(df),
        # Rang real de quota: clients sense internet paguen molt menys (evita perfils impossibles).
        "rang_quota": {sense: (float(np.floor(q.min())), float(np.ceil(q.max())))
                       for sense, q in quota},
        "cph_total": ajustar_cox(X),
        "cph_tr": cph_tr,
        "cph_e": cph_e,
        "columnes": list(X.columns.drop(["duration", "event"])),
        "cindex": {"train": c_index(cph_tr, X_tr), "test": c_index(cph_tr, X_te),
                   "test_estratificat": c_index(cph_e, preparar_estratificat(te))},
        "ph": comprovar_ph(cph_tr, X_tr),
        "ph_e": comprovar_ph(cph_e, preparar_estratificat(tr)),
        "hz": hazard_mensual(df),
    }


@st.cache_data(show_spinner="Calculant puntuacions…")
def scores(dir_outputs: str) -> pd.DataFrame:
    ruta = Path(dir_outputs) / "scores_supervivencia.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    return generar_scores(models()["df"])


def pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def dec(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def pval(p: float) -> str:
    return "< 0,001" if p < 0.001 else dec(p)


M = models()
df = M["df"]

# ---------- Capçalera ----------
with st.sidebar:
    st.title("Supervivència i retenció")
    st.caption("Cas pràctic · Telco Customer Churn")
    st.markdown(
        "**Temps:** mesos d'antiguitat (`tenure`)  \n"
        "**Esdeveniment:** baixa del client (`Churn = Yes`)  \n"
        "**Censura:** clients encara actius (`Churn = No`)"
    )
    st.divider()
    st.caption(f"{len(df):,} clients · {M['n_exclosos']} registres amb antiguitat 0 exclosos"
               .replace(",", "."))

tabs = st.tabs(["Resum", "Kaplan-Meier", "Model de Cox", "Risc temporal",
                "Simulador de client", "Risc × valor", "Classificació"])

# ---------- 1. Resum ----------
with tabs[0]:
    km_g = ajustar_km(df)["Global"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients analitzats", f"{len(df):,}".replace(",", "."))
    c2.metric("Taxa d'abandonament", pct(df["event"].mean()))
    c3.metric("Clients censurats (actius)", pct(1 - df["event"].mean()))
    c4.metric("Continuen al cap de 12 mesos", pct(float(km_g.predict(12))))
    st.plotly_chart(fig_km(ajustar_km(df), "Supervivència global dels clients (Kaplan-Meier)"),
                    **GRAFIC)
    st.caption("La mediana global no s'assoleix: més de la meitat dels clients continuen actius "
               "als 72 mesos. Per això es llegeix S(t) en horitzons concrets.")

# ---------- 2. Kaplan-Meier ----------
with tabs[1]:
    c1, c2 = st.columns([3, 1])
    var = c1.selectbox("Segmentar per", VARIABLES_KM, format_func=lambda v: NOMS_VARIABLES[v])
    interval = c2.toggle("Interval de confiança 95 %", value=True)
    fitters = ajustar_km(df, var)
    st.plotly_chart(fig_km(fitters, f"Supervivència per: {NOMS_VARIABLES[var].lower()}", interval),
                    **GRAFIC)
    lr = logrank(df, var)
    med = taula_medianes(fitters)
    med["grup"] = med["grup"].map(traduir)
    ordre = med.sort_values("S_12m")
    signe = "" if lr["p_valor"] < 0.001 else "= "
    st.markdown(f"**Test log-rank:** χ² = {dec(lr['estadistic'], 1)} · p {signe}{pval(lr['p_valor'])}")
    st.markdown(conclusio_logrank(lr["p_valor"], NOMS_VARIABLES[var].lower()))
    if lr["p_valor"] < 0.05:
        st.markdown(frase_grups(ordre["grup"].iloc[-1], ordre["grup"].iloc[0]))
    for c in ["S_6m", "S_12m", "S_24m"]:
        med[c] = med[c].map(pct)
    med["mediana"] = med["mediana"].map(text_mediana)
    st.dataframe(med.rename(columns={"grup": "Grup", "n": "Clients", "mediana": "Mediana (mesos)",
                                     "S_6m": "S(6)", "S_12m": "S(12)", "S_24m": "S(24)"}),
                 hide_index=True, width="stretch")
    st.caption("Kaplan-Meier no extrapola: només descriu fins al mes 72, l'antiguitat màxima "
               "del dataset.")

# ---------- 3. Model de Cox ----------
with tabs[2]:
    c1, c2, c3 = st.columns([2, 1, 1])
    estrat = c1.toggle("Estratificat per tipus de contracte", value=False,
                       help="Permet una funció de risc base diferent per a cada contracte, "
                            "perquè el contracte viola el supòsit de riscos proporcionals.")
    c2.metric("C-index (test)", f"{M['cindex']['test']:.3f}".replace(".", ","),
              help="Avaluat amb el 20 % de clients reservats que el model no ha vist mai: és la "
                   "mesura honesta de com ordenaria clients nous. Indica el percentatge de parells "
                   "de clients en què el model dona més risc al que marxa abans (0,5 = atzar; 1 = perfecte).")
    c3.metric("C-index (train)", f"{M['cindex']['train']:.3f}".replace(".", ","),
              help="Avaluat amb el 80 % de clients amb què s'ha ajustat el model. Es mostra per "
                   "comparar-lo amb el de test: si fos molt més alt, el model estaria sobreajustat.")
    cph = M["cph_e"] if estrat else M["cph_tr"]
    titol = "Hazard ratios — Cox estratificat per contracte" if estrat else "Hazard ratios del model de Cox"
    st.plotly_chart(fig_forest(taula_hazard_ratios(cph), titol + " (IC 95 %)", NOMS_COVARIABLES),
                    **GRAFIC)
    st.caption("Escala logarítmica: els hazard ratios són multiplicacions del risc, i en aquesta "
               "escala ×2 i ÷2 queden a la mateixa distància de l'1.")
    with st.expander("Taula dels hazard ratios amb interpretació"):
        hr = taula_hazard_ratios(cph)
        taula_hr = pd.DataFrame({
            "Variable": hr["variable"].map(NOMS_COVARIABLES),
            "HR": hr["HR"].map(lambda v: dec(v, 2)),
            "IC 95 %": [f"{dec(a, 2)} – {dec(b, 2)}" for a, b in zip(hr["IC_inf"], hr["IC_sup"])],
            "p-valor": hr["p_valor"].map(pval),
            "Interpretació": [interpretar_hr(h, p) for h, p in zip(hr["HR"], hr["p_valor"])],
        })
        st.dataframe(taula_hr, hide_index=True, width="stretch")
        n_sig = int((hr["p_valor"] < 0.05).sum())
        st.caption(f"{n_sig} de {len(hr)} variables tenen un efecte significatiu (p < 0,05), mantenint "
                   "la resta constants. «Significatiu» vol dir que, si la variable no tingués cap efecte, "
                   "seria molt improbable (< 5 %) observar un HR tan allunyat de l'1.")
    with st.expander("Comprovació del supòsit de riscos proporcionals (residus de Schoenfeld)"):
        ph = (M["ph_e"] if estrat else M["ph"]).copy()
        ph["variable"] = ph["variable"].map(NOMS_COVARIABLES)
        ph["Interpretació"] = ph["p_valor"].map(interpretar_ph)
        n_viola = int(ph["viola"].sum())
        ph["viola"] = ph["viola"].map({True: "Sí", False: "No"})
        ph["p_valor"] = ph["p_valor"].map(pval)
        st.dataframe(ph.rename(columns={"variable": "Variable", "p_valor": "p-valor",
                                        "viola": "Viola el supòsit (p < 0,05)"}),
                     hide_index=True, width="stretch")
        st.markdown(f"**Conclusió:** {n_viola} de {len(ph)} variables violen el supòsit. Per a "
                    "aquestes, el HR és un efecte *mitjà* al llarg del temps, no un efecte constant.")
        ci_e = f"{M['cindex']['test_estratificat']:.3f}".replace(".", ",")
        if estrat:
            st.caption("Taula del model estratificat. Estratificar només corregeix la violació del "
                       "contracte: la quota mensual, el pagament automàtic, el suport tècnic i altres "
                       "continuen violant el supòsit (efectes que canvien amb el temps), cosa que "
                       f"queda com a limitació. El seu C-index ({ci_e}) no és directament comparable, "
                       "perquè el contracte passa a la funció de risc base.")
        else:
            st.caption("Diverses variables violen el supòsit, la més forta la quota mensual. "
                       "Activant el model estratificat es corregeix la del contracte; les altres "
                       "(efectes que canvien amb el temps) queden com a limitació.")

# ---------- 4. Risc temporal ----------
with tabs[3]:
    hz = M["hz"]
    critics = mesos_critics(hz)
    contractes = st.multiselect("Comparar per tipus de contracte (opcional)",
                                sorted(df["Contract"].unique()), format_func=traduir)
    if contractes:
        hz_c = hazard_mensual(df[df["Contract"].isin(contractes)], "Contract")
        fig = fig_hazard(hz_c[hz_c["en_risc"] >= 50], [], "Taxa d'abandonament mensual per contracte")
    else:
        fig = fig_hazard(hz, critics, "Taxa d'abandonament per mes d'antiguitat")
    st.plotly_chart(fig, **GRAFIC)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baixes el 1r mes", pct(float(hz.loc[0, "hazard"])))
    c2.metric("Taxa mensual mitjana (mesos 6–60)",
              pct(float(hz[(hz.mes >= 6) & (hz.mes <= 60)]["hazard"].mean())))
    c3.metric("Mesos crítics (global)", ", ".join(map(str, critics)))
    st.caption("Risc en forma de U: finestra d'incorporació (mesos 1–5) i repunt al final de "
               "l'horitzó (66–72). Només es consideren mesos amb ≥ 50 clients en risc.")

# ---------- 5. Simulador de client ----------
OPCIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}
PERFILS_INICIALS = {
    "A": {"Contract": "Month-to-month", "InternetService": "Fiber optic",
          "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0},
    "B": {"Contract": "Two year", "InternetService": "DSL",
          "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 60.0},
}


def controls_perfil(clau: str) -> dict:
    ini = PERFILS_INICIALS[clau]
    p = {}
    for var in ["Contract", "InternetService", "PaymentMethod"]:
        p[var] = st.selectbox(NOMS_VARIABLES[var], OPCIONS[var], index=OPCIONS[var].index(ini[var]),
                              format_func=traduir, key=f"{clau}_{var}")
    sense_internet = p["InternetService"] == "No"
    p["OnlineSecurity"] = "Yes" if st.checkbox("Seguretat en línia", key=f"{clau}_sec",
                                               disabled=sense_internet) and not sense_internet else "No"
    p["TechSupport"] = "Yes" if st.checkbox("Suport tècnic", key=f"{clau}_ts",
                                            disabled=sense_internet) and not sense_internet else "No"
    p["PaperlessBilling"] = "Yes" if st.checkbox("Factura electrònica", value=True,
                                                 key=f"{clau}_pb") else "No"
    p["Partner"] = "Yes" if st.checkbox("Té parella", key=f"{clau}_pa") else "No"
    p["Dependents"] = "Yes" if st.checkbox("Té persones a càrrec", key=f"{clau}_de") else "No"
    p["SeniorCitizen"] = int(st.checkbox("Client sènior", key=f"{clau}_se"))
    mn, mx = M["rang_quota"][sense_internet]
    # Clau diferent per mode: en canviar el servei d'internet el control es reinicia dins el rang real.
    p["MonthlyCharges"] = st.slider("Quota mensual", mn, mx,
                                    min(max(ini["MonthlyCharges"], mn), mx), 1.0,
                                    format="%.0f €",
                                    key=f"{clau}_mc_{'sense' if sense_internet else 'amb'}_internet",
                                    help="Rang de quotes observat al dataset per a aquest tipus "
                                         "de servei d'internet.")
    return p


with tabs[4]:
    st.markdown("Configura dos perfils de client nou i compara la seva corba de supervivència "
                "prevista pel model de Cox.")
    cA, cB, cG = st.columns([1, 1, 2.4])
    corbes, resultats = {}, {}
    temps = np.arange(0, 73)
    for clau, col in [("A", cA), ("B", cB)]:
        with col:
            st.subheader(f"Perfil {clau}")
            perfil = controls_perfil(clau)
        fila = codificar_perfil(perfil, M["columnes"])
        s = M["cph_total"].predict_survival_function(fila, times=temps).iloc[:, 0]
        corbes[f"Perfil {clau}"] = s
        resultats[clau] = {"S6": float(s.loc[6]), "S12": float(s.loc[12]),
                           "clv": float(s.loc[1:24].sum() * perfil["MonthlyCharges"])}
    with cG:
        st.plotly_chart(fig_corbes_client(corbes, "Supervivència prevista per perfil"), **GRAFIC)
        taula = pd.DataFrame({
            "": ["Continua als 6 mesos", "Continua als 12 mesos", "CLV esperat a 24 mesos"],
            "Perfil A": [pct(resultats["A"]["S6"]), pct(resultats["A"]["S12"]), euros(resultats["A"]["clv"])],
            "Perfil B": [pct(resultats["B"]["S6"]), pct(resultats["B"]["S12"]), euros(resultats["B"]["clv"])],
        })
        st.dataframe(taula, hide_index=True, width="stretch")
        st.caption("La corba arriba fins als 72 mesos perquè és l'antiguitat màxima observada al "
                   "dataset: el model de Cox estima el risc base només on hi ha dades. Més enllà, "
                   "qualsevol valor seria una extrapolació sense base empírica.")

# ---------- 6. Risc × valor ----------
with tabs[5]:
    incloure = st.toggle(
        "Incloure també els clients que ja han marxat (comparació)", value=False,
        key="incloure_baixes",
        help="Per defecte només es mostren els clients actius (censurats), que són els únics que "
             "es poden retenir. Activa-ho per veure com canvien la matriu i els segments.")
    sc = scores(str(DIR_OUTPUTS))
    tots = sc.merge(df[["customerID", "MonthlyCharges", "event"]], on="customerID")
    actius = tots[tots["event"] == 0]
    base = tots if incloure else actius
    if incloure:
        st.warning("Inclou clients que ja han marxat: el seu risc i CLV són hipotètics (com si "
                   "encara fossin clients) i el CLV queda inflat perquè pagaven quotes més altes. "
                   "Útil només per comparar; les decisions s'han de prendre amb els clients actius.")
    font = "risc de baixa a 12 mesos (model de Cox)"
    fig = fig_risc_valor(base, base["MonthlyCharges"],
                         marxat=(base["event"] == 1) if incloure else None)
    st.plotly_chart(fig, **GRAFIC)
    poblacio = "Tots els clients (actius + baixes)" if incloure else "Clients actius"
    st.caption(f"{poblacio} · eix horitzontal: {font} · n = {len(base)}")

    def perfil_segments(d: pd.DataFrame) -> pd.DataFrame:
        return (d.groupby("segment_risc")
                .agg(clients=("customerID", "size"), S_12m=("S_12m", "mean"),
                     clv=("clv_esperat_24m", "mean"))
                .reindex(["Alt", "Mitjà", "Baix"]))

    pa = perfil_segments(actius)
    seg = pd.DataFrame({"Segment de risc (Cox)": pa.index, "Clients actius": pa["clients"].values,
                        "Continuen a 12 mesos": pa["S_12m"].map(pct).values,
                        "CLV esperat 24m": pa["clv"].map(euros).values})
    if incloure:
        pt = perfil_segments(tots)
        seg["Clients (tots)"] = pt["clients"].values
        seg["Continuen a 12 mesos (tots)"] = pt["S_12m"].map(pct).values
        seg["CLV esperat 24m (tots)"] = pt["clv"].map(euros).values
        seg["Diferència CLV"] = (pt["clv"] - pa["clv"]).map(lambda v: "+" + euros(v)).values
    st.dataframe(seg, hide_index=True, width="stretch")

# ---------- 7. Classificació ----------
@st.cache_resource(show_spinner="Preparant les dades de classificació…")
def dades_classificacio() -> dict:
    return preparar_dades(models()["df"])


@st.cache_resource(show_spinner="Entrenant els models de classificació…")
def models_classificacio(tractament: str) -> dict:
    d = dades_classificacio()
    ms = ajustar_tots(tractament, d)
    return {"models": ms, "probs": {n: probabilitats(m, d["X_te"]) for n, m in ms.items()}}


@st.cache_data(show_spinner="Calculant la importància de les variables…")
def importancia_classificacio(nom: str, tractament: str) -> pd.DataFrame:
    d = dades_classificacio()
    model = models_classificacio(tractament)["models"][nom]
    return importancia_variables(model, d["X_te"], d["y_te"], repeticions=3)


OPCIONS_CL = {
    **OPCIONS,
    "gender": ["Female", "Male"],
}
SERVEIS_CL = [("OnlineSecurity", "Seguretat en línia"), ("TechSupport", "Suport tècnic"),
              ("OnlineBackup", "Còpia de seguretat en línia"),
              ("DeviceProtection", "Protecció de dispositius"),
              ("StreamingTV", "Televisió en streaming"), ("StreamingMovies", "Pel·lícules en streaming")]

with tabs[6]:
    st.markdown("Predir **qui** marxarà (sí o no) a partir de les característiques del client. Els "
                "models aprenen amb el 80 % dels clients i s'avaluen amb el 20 % restant, que no han "
                "vist mai: els **mateixos clients de prova** que el model de Cox.")
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    nom_model = c1.selectbox("Model", MODELS, key="cl_model",
                             help="Quatre maneres d'aprendre a predir la baixa. A sota hi ha "
                                  "l'explicació del model triat.")
    tractament = c2.selectbox("Tractament del desequilibri de classes", TRACTAMENTS, index=1,
                              key="cl_tractament",
                              help="Només marxa el 27 % dels clients: sense tractament, els models "
                                   "tendeixen a dir «no marxa». " + " ".join(
                                       f"{k}: {v}" for k, v in DESCRIPCIONS_TRACTAMENTS.items()))
    llindar = c3.slider("Llindar de decisió", 0.05, 0.95, 0.50, 0.05, key="cl_llindar",
                        help=AJUDA_METRIQUES["llindar"])
    st.info(DESCRIPCIONS_MODELS[nom_model])
    st.caption(f"**Tractament «{tractament}»:** {DESCRIPCIONS_TRACTAMENTS[tractament]}")
    if tractament == "SMOTE" and not smote_disponible():
        st.warning("No s'ha pogut carregar la llibreria de SMOTE (imbalanced-learn) en aquest "
                   "ordinador: es mostren els resultats amb **pesos de classe**. Sol passar a "
                   "Windows 11 quan el Control intel·ligent d'aplicacions bloqueja una DLL.")
        tractament = "Pesos de classe"

    D = dades_classificacio()
    R = models_classificacio(tractament)
    p = R["probs"][nom_model]
    mt = avaluar(D["y_te"], p, llindar)

    # Rendiment del model seleccionat
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("AUC", dec(mt["auc"]), help=AJUDA_METRIQUES["auc"])
    k2.metric("Recall", pct(mt["recall"]), help=AJUDA_METRIQUES["recall"])
    k3.metric("Precision", pct(mt["precision"]), help=AJUDA_METRIQUES["precision"])
    k4.metric("F1", dec(mt["f1"]), help=AJUDA_METRIQUES["f1"])
    k5.metric("Exactitud", pct(mt["exactitud"]), help=AJUDA_METRIQUES["exactitud"])
    st.markdown(frase_metriques(mt))
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_confusio(mt, f"Matriu de confusió · {nom_model}"), **GRAFIC)
        st.caption("Compara la predicció amb el que va passar de veritat als 1.407 clients de prova. "
                   "Blau = encerts; taronja = errors.")
    with g2:
        rocs = {n: corba_roc(D["y_te"], pr) for n, pr in R["probs"].items()}
        aucs = {n: avaluar(D["y_te"], pr)["auc"] for n, pr in R["probs"].items()}
        st.plotly_chart(fig_roc(rocs, aucs, nom_model, "Corbes ROC (tots els llindars)"), **GRAFIC)
        st.caption("Cada punt de la corba és un llindar diferent. Com més amunt i a l'esquerra, "
                   "millor: més baixes detectades amb menys falses alarmes. L'AUC és l'àrea sota "
                   "la corba.")

    # Interpretació
    st.subheader("Per què decideix així?")
    model_sel = R["models"][nom_model]
    if nom_model == "Regressió logística":
        ors = odds_ratios(model_sel, D["columnes"])
        st.plotly_chart(fig_odds_ratios(ors, "Odds ratios de la regressió logística"), **GRAFIC)
        st.caption("Mantenint la resta de variables iguals, un odds ratio < 1 (blau) redueix la "
                   "probabilitat relativa de marxar i un > 1 (taronja) l'augmenta. Els contractes "
                   "es comparen amb el mensual i el pagament amb el xec postal, com al model de Cox.")
        with st.expander("Taula dels odds ratios amb interpretació"):
            st.dataframe(pd.DataFrame({
                "Variable": ors["nom"], "Odds ratio": ors["OR"].map(lambda v: dec(v, 2)),
                "Interpretació": ors["OR"].map(interpretar_or)}), hide_index=True, width="stretch")
            st.caption("La quota mensual surt amb poc efecte perquè és pràcticament la suma dels "
                       "serveis contractats, que ja són al model: l'efecte es reparteix entre ells.")
    elif nom_model == "Arbre de decisió":
        est = estructura_arbre(model_sel, D["columnes"], D["X_tr"], D["y_tr"])
        st.plotly_chart(fig_arbre(est, "Arbre de decisió: preguntes que ha après el model"), **GRAFIC)
        st.caption("Es llegeix de dalt a baix. Cada caixa és una pregunta; les fulles (a baix) donen la "
                   "probabilitat de baixa que assigna el model (taronja = risc alt). Passant el ratolí "
                   "per sobre es veu quants clients reals hi ha i quants van marxar. Amb pesos o SMOTE, "
                   "la probabilitat del model és més alta que la taxa real, perquè s'ha entrenat "
                   "donant més pes a les baixes.")
    else:
        st.plotly_chart(fig_importancia(importancia_classificacio(nom_model, tractament),
                                        f"Variables més importants · {nom_model}"), **GRAFIC)
        st.caption("Importància per permutació: es barregen a l'atzar els valors d'una variable i es "
                   "mira quant empitjora l'AUC. Si cau molt, el model en depenia. Indica **quant** "
                   "pesa una variable, però no cap a on (per a això, la regressió logística).")
    if nom_model in ("Regressió logística", "Arbre de decisió"):
        with st.expander("Importància de les variables (comparable entre els quatre models)"):
            st.plotly_chart(fig_importancia(importancia_classificacio(nom_model, tractament),
                                            f"Variables més importants · {nom_model}"), **GRAFIC)

    # Comparativa
    st.subheader("Comparativa dels quatre models")
    comp = taula_comparativa(R["models"], D, llindar, 0.10)
    millor = comp.loc[comp["auc"].idxmax(), "model"]
    st.dataframe(pd.DataFrame({
        "Model": comp["model"], "AUC": comp["auc"].map(dec), "Recall": comp["recall"].map(pct),
        "Precision": comp["precision"].map(pct), "F1": comp["f1"].map(dec),
        "Exactitud": comp["exactitud"].map(pct), "Lift (10 %)": comp["lift"].map(lambda v: dec(v, 2)),
    }), hide_index=True, width="stretch")
    st.caption(f"Amb el tractament «{tractament}» i llindar {dec(llindar, 2)}. Millor AUC: "
               f"**{millor}**. Les diferències d'AUC entre models són petites: les variables "
               "importants (antiguitat, contracte, internet) les capten tots.")

    # Campanya
    st.subheader("De la predicció a l'acció: campanya de retenció")
    pct_c = st.slider("Percentatge de clients a contactar", 5, 50, 10, 5, format="%d %%",
                      key="cl_pct", help="L'empresa ordena els clients de més a menys risc segons el "
                                         "model i truca als primers.")
    cmp = campanya(D["y_te"], p, pct_c / 100)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Clients contactats", f"{cmp['contactats']:,}".replace(",", "."))
    q2.metric("Baixes detectades", f"{cmp['baixes_detectades']} de {cmp['baixes_totals']}")
    q3.metric("Baixes capturades", pct(cmp["pct_capturat"]),
              help="Percentatge de totes les baixes que cauen dins dels clients contactats.")
    q4.metric("Lift", dec(cmp["lift"], 2), help=AJUDA_METRIQUES["lift"])
    st.markdown(f"Trucant al **{pct_c} %** de clients amb més risc ({cmp['contactats']}), es troben "
                f"**{cmp['baixes_detectades']}** dels {cmp['baixes_totals']} que marxaran: "
                f"**{dec(cmp['lift'], 1)} vegades** més que triant-los a l'atzar (on se'n trobarien "
                f"≈ {round(cmp['contactats'] * cmp['taxa_base'])}).")
    guanys = {n: corba_guanys(D["y_te"], pr) for n, pr in R["probs"].items()}
    st.plotly_chart(fig_guanys(guanys, nom_model, pct_c / 100, "Corba de guanys"), **GRAFIC)
    st.caption("Com més s'allunya la corba de la diagonal (atzar), més útil és el model per "
               "prioritzar trucades amb un pressupost limitat.")

    # Simulador
    st.subheader("Simulador: prova un client")
    st.markdown("Configura un client i mira quina probabilitat de baixa li dona cada model.")
    s1, s2, s3 = st.columns(3)
    pc = {}
    with s1:
        for var in ["Contract", "InternetService", "PaymentMethod"]:
            pc[var] = st.selectbox(NOMS_VARIABLES[var], OPCIONS_CL[var], format_func=traduir,
                                   key=f"cl_{var}",
                                   index=OPCIONS_CL[var].index(PERFILS_INICIALS["A"][var]))
        pc["tenure"] = st.slider("Antiguitat (mesos)", 1, 72, 3, key="cl_tenure")
        sense_int = pc["InternetService"] == "No"
        mn, mx = M["rang_quota"][sense_int]
        pc["MonthlyCharges"] = st.slider("Quota mensual", mn, mx, min(max(85.0, mn), mx), 1.0,
                                         format="%.0f €",
                                         key=f"cl_mc_{'sense' if sense_int else 'amb'}_internet")
    with s2:
        for var, nom in SERVEIS_CL:
            pc[var] = "Yes" if (st.checkbox(nom, key=f"cl_{var}", disabled=sense_int)
                                and not sense_int) else "No"
        pc["PhoneService"] = "Yes" if st.checkbox("Té telefonia", value=True, key="cl_ph") else "No"
        pc["MultipleLines"] = "Yes" if (st.checkbox("Diverses línies", key="cl_ml",
                                                    disabled=pc["PhoneService"] == "No")
                                        and pc["PhoneService"] == "Yes") else "No"
    with s3:
        pc["PaperlessBilling"] = "Yes" if st.checkbox("Factura electrònica", value=True,
                                                      key="cl_pb") else "No"
        pc["Partner"] = "Yes" if st.checkbox("Té parella", key="cl_pa") else "No"
        pc["Dependents"] = "Yes" if st.checkbox("Té persones a càrrec", key="cl_de") else "No"
        pc["SeniorCitizen"] = int(st.checkbox("Client sènior", key="cl_se"))
        pc["gender"] = st.radio("Gènere", ["Female", "Male"], horizontal=True, key="cl_gender",
                                format_func=lambda v: "Dona" if v == "Female" else "Home")
    fila_cl = codificar_perfil_classificacio(pc, D["columnes"])
    probs_perfil = {n: float(probabilitats(m, fila_cl)[0]) for n, m in R["models"].items()}
    st.plotly_chart(fig_probabilitats_perfil(probs_perfil, llindar, nom_model,
                                             "Probabilitat de baixa segons cada model"), **GRAFIC)
    risc = "en risc" if probs_perfil[nom_model] >= llindar else "fora de risc"
    st.markdown(f"Segons el model **{nom_model}**, aquest client té una probabilitat de baixa del "
                f"**{pct(probs_perfil[nom_model])}**: amb el llindar {dec(llindar, 2)} queda "
                f"**{risc}**.")
    arbre = R["models"]["Arbre de decisió"]
    est_cl = estructura_arbre(arbre, D["columnes"], D["X_tr"], D["y_tr"])
    with st.expander("Camí que segueix aquest client a l'arbre de decisió",
                     expanded=nom_model == "Arbre de decisió"):
        passos = cami_arbre(arbre, fila_cl, est_cl)
        st.markdown("\n".join(f"{i}. {pas['pregunta']} → **{pas['resposta']}**"
                              for i, pas in enumerate(passos, 1)))
        st.plotly_chart(fig_arbre(est_cl, "Recorregut del client a l'arbre",
                                  nodes_cami(arbre, fila_cl)), **GRAFIC)

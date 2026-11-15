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
                "Simulador de client", "Risc × valor"])

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

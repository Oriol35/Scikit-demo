"""Gràfics Plotly dels models de classificació (mateix estil que `src/grafics.py`)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.grafics import (COLOR_PROTECTOR, COLOR_RISC, PALETA, SUPERFICIE, TEXT_PRIMARI,
                         TEXT_SECUNDARI, _estil, _rgba)

GRIS = "#b4b2a9"
# Llegenda dins del gràfic (cantonada inferior dreta, zona buida a les corbes ROC i de guanys).
LLEGENDA_DINS = dict(orientation="v", y=0.03, yanchor="bottom", x=0.98, xanchor="right",
                     bgcolor="rgba(252,252,251,0.85)", font=dict(size=11, color=TEXT_PRIMARI))


def _pct(x: float) -> str:
    return f"{x:.0%}".replace("%", " %")


def _milers(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fig_confusio(m: dict, titol: str) -> go.Figure:
    """Matriu de confusió 2×2: encerts en blau, errors en taronja."""
    files = ["Ha marxat", "No ha marxat"]          # realitat
    columnes = ["Predicció: marxarà", "Predicció: no marxarà"]
    valors = [[m["vp"], m["fn"]], [m["fp"], m["vn"]]]
    noms = [["Vertaders positius<br>(baixes detectades)", "Falsos negatius<br>(baixes escapades)"],
            ["Falsos positius<br>(falses alarmes)", "Vertaders negatius<br>(bé: no marxen)"]]
    encert = [[1, 0], [0, 1]]
    fig = go.Figure(go.Heatmap(
        z=encert, x=columnes, y=files, showscale=False, hoverinfo="skip", xgap=4, ygap=4,
        colorscale=[[0, _rgba(COLOR_RISC, 0.22)], [1, _rgba(COLOR_PROTECTOR, 0.22)]]))
    for i, fila in enumerate(files):
        for j, col in enumerate(columnes):
            fig.add_annotation(x=col, y=fila, showarrow=False, font=dict(color=TEXT_PRIMARI, size=13),
                               text=f"<b style='font-size:22px'>{_milers(valors[i][j])}</b><br>{noms[i][j]}")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_layout(height=420)
    fig = _estil(fig, titol, "", "Realitat")
    fig.update_layout(margin=dict(t=110, l=110))
    fig.update_yaxes(automargin=True)
    return fig


def fig_roc(corbes: dict[str, pd.DataFrame], aucs: dict[str, float], seleccionat: str,
            titol: str) -> go.Figure:
    """Corbes ROC dels models; el seleccionat en color i la resta en gris."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="A l'atzar (AUC = 0,5)",
                             line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"), hoverinfo="skip"))
    for nom, c in corbes.items():
        sel = nom == seleccionat
        etiqueta = f"{nom} (AUC = {aucs[nom]:.3f})".replace(".", ",")
        fig.add_trace(go.Scatter(
            x=c["fpr"], y=c["tpr"], mode="lines", name=etiqueta,
            line=dict(color=PALETA[0] if sel else GRIS, width=3 if sel else 1.3),
            customdata=c["llindar"].clip(upper=1),
            hovertemplate=(f"<b>{nom}</b><br>Falses alarmes = %{{x:.1%}}<br>Recall = %{{y:.1%}}"
                           "<br>Llindar = %{customdata:.2f}<extra></extra>")))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_layout(height=460)
    fig = _estil(fig, titol, "Falses alarmes (clients que no marxen marcats en risc)",
                 "Recall (baixes detectades)")
    return fig.update_layout(legend=LLEGENDA_DINS)


def fig_odds_ratios(taula: pd.DataFrame, titol: str) -> go.Figure:
    """Odds ratios de la regressió logística (eix logarítmic, com el forest plot del Cox)."""
    t = taula.sort_values("OR", ascending=False)
    colors = [COLOR_PROTECTOR if v < 1 else COLOR_RISC for v in t["OR"]]
    fig = go.Figure(go.Scatter(
        x=t["OR"], y=t["nom"], mode="markers",
        marker=dict(size=11, color=colors, line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="<b>%{y}</b><br>Odds ratio = %{x:.2f}<extra></extra>", showlegend=False))
    fig.add_vline(x=1, line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"))
    ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
    fig.update_xaxes(type="log", range=[np.log10(0.15), np.log10(4)], tickvals=ticks,
                     ticktext=[f"{v:g}".replace(".", ",") for v in ticks])
    fig.update_layout(height=max(420, 30 * len(t) + 140))
    fig = _estil(fig, titol, "Odds ratio (escala log.)  ·  < 1 redueix el risc de baixa  ·  > 1 l'augmenta", "")
    fig.update_yaxes(automargin=True)
    return fig


def fig_importancia(taula: pd.DataFrame, titol: str, n: int = 10) -> go.Figure:
    """Importància per permutació: quant cau l'AUC si es barreja cada variable."""
    t = taula.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["importancia"], y=t["nom"], orientation="h", marker_color=PALETA[0],
        error_x=dict(type="data", array=t["desviacio"], color=TEXT_SECUNDARI, thickness=1.2, width=3),
        hovertemplate="<b>%{y}</b><br>L'AUC cau %{x:.3f}<extra></extra>"))
    fig.update_layout(height=max(380, 34 * len(t) + 140), bargap=0.35)
    fig = _estil(fig, titol, "Caiguda de l'AUC en barrejar la variable (més = més important)", "")
    fig.update_yaxes(automargin=True)
    return fig


def _color_risc(p: float) -> str:
    """De blau (risc baix) a taronja (risc alt)."""
    a = np.array([int(COLOR_PROTECTOR[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(COLOR_RISC[i:i + 2], 16) for i in (1, 3, 5)])
    r, g, bl = (a + (b - a) * float(np.clip(p, 0, 1))).astype(int)
    return f"rgba({r},{g},{bl},0.25)"


def fig_arbre(estructura: pd.DataFrame, titol: str, cami: list[int] | None = None) -> go.Figure:
    """Dibuix de l'arbre: preguntes als nodes i probabilitat de baixa a les fulles.

    Les fulles es reparteixen d'esquerra a dreta i cada pregunta queda centrada sobre els seus
    fills. Si es dona `cami` (llista de nodes), es ressalta el recorregut d'un client.
    """
    t = estructura.set_index("node")
    fills = {i: [] for i in t.index}
    for i, fila in t.iterrows():
        if fila["pare"] != -1:
            fills[fila["pare"]].append(i)
    for i in fills:  # esquerra abans que dreta
        fills[i].sort(key=lambda f: 0 if t.loc[f, "costat"] == "esquerra" else 1)
    x, comptador = {}, [0]

    def posar(i):
        if not fills[i]:
            x[i] = comptador[0]
            comptador[0] += 1
        else:
            for f in fills[i]:
                posar(f)
            x[i] = float(np.mean([x[f] for f in fills[i]]))
    posar(0)
    cami = set(cami or [])
    fig = go.Figure()
    for i, fila in t.iterrows():
        if fila["pare"] == -1:
            continue
        pare = t.loc[fila["pare"]]
        x0, y0, x1, y1 = x[fila["pare"]], -pare["profunditat"], x[i], -fila["profunditat"]
        actiu = i in cami and fila["pare"] in cami
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                 showlegend=False,
                                 line=dict(color=COLOR_RISC if actiu else GRIS, width=3 if actiu else 1)))
        resposta = "Sí" if fila["costat"] == pare["branca_si"] else "No"
        fig.add_annotation(x=x0 + (x1 - x0) * 0.35, y=y0 + (y1 - y0) * 0.35, text=resposta,
                           showarrow=False, font=dict(size=11, color=TEXT_SECUNDARI),
                           bgcolor=SUPERFICIE)
    hover_x, hover_y, hover_text = [], [], []
    for i, fila in t.iterrows():
        if fila["fulla"]:
            # Fulles: text curt perquè n'hi ha moltes una al costat de l'altra.
            text = (f"<b>{_pct(fila['prob_model'])}</b><br>"
                    f"<span style='font-size:10px'>n = {_milers(fila['n'])}</span>")
            fons = _color_risc(fila["prob_model"])
        else:
            text = (f"<b>{fila['pregunta']}</b><br><span style='font-size:10px'>"
                    f"n = {_milers(fila['n'])} · marxen {_pct(fila['taxa_real'])}</span>")
            fons = SUPERFICIE
        fig.add_annotation(x=x[i], y=-fila["profunditat"], text=text, showarrow=False,
                           font=dict(size=11, color=TEXT_PRIMARI), bgcolor=fons, borderpad=4,
                           bordercolor=COLOR_RISC if i in cami else GRIS,
                           borderwidth=2 if i in cami else 1)
        hover_x.append(x[i])
        hover_y.append(-fila["profunditat"])
        cap = "Fulla" if fila["fulla"] else fila["pregunta"]
        hover_text.append(f"<b>{cap}</b><br>{_milers(fila['n'])} clients d'entrenament<br>"
                          f"Han marxat de veritat: {_pct(fila['taxa_real'])}<br>"
                          f"Probabilitat del model: {_pct(fila['prob_model'])}")
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", showlegend=False,
                             marker=dict(size=28, opacity=0), text=hover_text,
                             hovertemplate="%{text}<extra></extra>"))
    max_prof = int(t["profunditat"].max())
    fig.update_xaxes(visible=False, range=[-0.7, comptador[0] - 0.3])
    fig.update_yaxes(visible=False, range=[-max_prof - 0.5, 0.5])
    fig.update_layout(height=170 * (max_prof + 1), showlegend=False)
    fig = _estil(fig, titol, "", "")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def fig_guanys(corbes: dict[str, pd.DataFrame], seleccionat: str, pct: float,
               titol: str) -> go.Figure:
    """Corba de guanys: % de clients contactats (de més a menys risc) → % de baixes capturades."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Triant clients a l'atzar",
                             line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"), hoverinfo="skip"))
    for nom, c in corbes.items():
        sel = nom == seleccionat
        fig.add_trace(go.Scatter(
            x=c["pct_contactat"], y=c["pct_capturat"], mode="lines", name=nom,
            line=dict(color=PALETA[0] if sel else GRIS, width=3 if sel else 1.3),
            hovertemplate=(f"<b>{nom}</b><br>Contactant el %{{x:.0%}} dels clients<br>"
                           "es capturen el %{y:.0%} de les baixes<extra></extra>")))
    c = corbes[seleccionat]
    k = int(np.searchsorted(c["pct_contactat"].to_numpy(), pct))
    k = min(k, len(c) - 1)
    fig.add_trace(go.Scatter(
        x=[c["pct_contactat"].iloc[k]], y=[c["pct_capturat"].iloc[k]], mode="markers",
        name="Campanya seleccionada", marker=dict(size=13, color=COLOR_RISC, symbol="diamond",
                                                  line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="Campanya: %{x:.0%} contactats → %{y:.0%} de les baixes<extra></extra>"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_layout(height=460)
    fig = _estil(fig, titol, "Clients contactats (ordenats de més a menys risc)",
                 "Baixes capturades")
    return fig.update_layout(legend=LLEGENDA_DINS)


def fig_probabilitats_perfil(probs: dict[str, float], llindar: float, seleccionat: str,
                             titol: str) -> go.Figure:
    """Probabilitat de baixa d'un client simulat segons cada model, amb el llindar marcat."""
    noms = list(probs)[::-1]
    valors = [probs[n] for n in noms]
    colors = [(COLOR_RISC if v >= llindar else COLOR_PROTECTOR) for v in valors]
    opac = [1 if n == seleccionat else 0.45 for n in noms]
    fig = go.Figure(go.Bar(
        x=valors, y=noms, orientation="h", marker=dict(color=colors, opacity=opac),
        text=[_pct(v) for v in valors], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probabilitat de baixa = %{x:.1%}<extra></extra>"))
    fig.add_vline(x=llindar, line=dict(color=TEXT_SECUNDARI, width=1.5, dash="dash"),
                  annotation_text=f"Llindar {llindar:.2f}".replace(".", ","),
                  annotation_position="top")
    fig.update_xaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(height=300, bargap=0.35)
    fig = _estil(fig, titol, "Probabilitat de baixa", "")
    fig.update_yaxes(automargin=True)
    return fig

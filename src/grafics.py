"""Gràfics Plotly compartits pels notebooks i el dashboard (etiquetes en català)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.preparacio import ARREL

DIR_FIGURES = ARREL / "figures"

# Paleta categòrica validada (ordre fix, mai ciclat) i tokens de text/superfície.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARI = "#0b0b0b"
TEXT_SECUNDARI = "#52514e"
SUPERFICIE = "#fcfcfb"
REIXA = "#e6e5e0"
COLOR_PROTECTOR = PALETA[0]
COLOR_RISC = PALETA[1]

TRADUCCIONS = {
    "Month-to-month": "Mensual", "One year": "1 any", "Two year": "2 anys",
    "DSL": "DSL", "Fiber optic": "Fibra òptica", "No": "No", "Yes": "Sí",
    "Electronic check": "Xec electrònic", "Mailed check": "Xec postal",
    "Bank transfer (automatic)": "Transferència (automàtica)",
    "Credit card (automatic)": "Targeta (automàtica)",
    "0": "No", "1": "Sí", "Global": "Tots els clients",
}


def traduir(valor: str) -> str:
    return TRADUCCIONS.get(str(valor), str(valor))


def _rgba(hex_color: str, alfa: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


def esglaons(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordenades explícites d'una funció esglaonada (valor constant fins al temps següent)."""
    t, v = np.asarray(t), np.asarray(v)
    x = np.concatenate([[t[0]], np.repeat(t[1:], 2)])
    y = np.concatenate([np.repeat(v[:-1], 2), [v[-1]]])
    return x, y


def _estil(fig: go.Figure, titol: str, x: str, y: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=titol, font=dict(size=18, color=TEXT_PRIMARI), x=0, xanchor="left"),
        paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT_SECUNDARI),
        margin=dict(l=60, r=30, t=95, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(color=TEXT_PRIMARI)),
        hoverlabel=dict(bgcolor="white", font_color=TEXT_PRIMARI),
        separators=",.",  # format català: coma decimal, punt de milers
    )
    eix = dict(gridcolor=REIXA, zeroline=False, linecolor=REIXA, title_font=dict(color=TEXT_SECUNDARI))
    fig.update_xaxes(title_text=x, **eix)
    fig.update_yaxes(title_text=y, **eix)
    return fig


def fig_km(fitters: dict, titol: str, interval: bool = True) -> go.Figure:
    """Corbes de Kaplan-Meier, una per grup (color fix per ordre del grup)."""
    fig = go.Figure()
    for i, (nom, kmf) in enumerate(fitters.items()):
        color = PALETA[i % len(PALETA)]
        etiqueta = traduir(nom)
        sf = kmf.survival_function_.iloc[:, 0]
        if interval:
            ci = kmf.confidence_interval_survival_function_
            t = ci.index.to_numpy()
            xs, ys_sup = esglaons(t, ci.iloc[:, 1].to_numpy())
            _, ys_inf = esglaons(t, ci.iloc[:, 0].to_numpy())
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs, xs[::-1]]), y=np.concatenate([ys_sup, ys_inf[::-1]]),
                fill="toself", fillcolor=_rgba(color, 0.15), line=dict(width=0),
                hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=sf.index, y=sf.values, mode="lines", name=etiqueta, line_shape="hv",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{etiqueta}</b><br>Mes %{{x}}<br>S(t) = %{{y:.1%}}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(range=[0, 72], dtick=12)
    fig.update_layout(hovermode="x unified")
    return _estil(fig, titol, "Mesos d'antiguitat", "Probabilitat de continuar sent client")


def fig_forest(taula_hr: pd.DataFrame, titol: str, noms: dict | None = None) -> go.Figure:
    """Forest plot dels hazard ratios amb IC 95 % (eix logarítmic)."""
    t = taula_hr.sort_values("HR", ascending=False)
    etiquetes = [(noms or {}).get(v, v) for v in t["variable"]]
    colors = [COLOR_PROTECTOR if hr < 1 else COLOR_RISC for hr in t["HR"]]
    fig = go.Figure(go.Scatter(
        x=t["HR"], y=etiquetes, mode="markers",
        marker=dict(size=11, color=colors, line=dict(color=SUPERFICIE, width=2)),
        error_x=dict(type="data", symmetric=False, array=t["IC_sup"] - t["HR"],
                     arrayminus=t["HR"] - t["IC_inf"], color=TEXT_SECUNDARI, thickness=1.5, width=0),
        customdata=np.stack([t["IC_inf"], t["IC_sup"], t["p_valor"]], axis=1),
        hovertemplate=("<b>%{y}</b><br>HR = %{x:.2f}<br>IC 95 %: %{customdata[0]:.2f} – "
                       "%{customdata[1]:.2f}<br>p = %{customdata[2]:.3g}<extra></extra>"),
        showlegend=False))
    fig.add_vline(x=1, line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"))
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5]
    fig.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{v:g}".replace(".", ",") for v in ticks])
    fig.update_layout(height=max(420, 34 * len(t) + 140))
    fig = _estil(fig, titol, "Hazard ratio (escala log.)  ·  < 1 redueix el risc de baixa  ·  > 1 l'augmenta", "")
    fig.update_yaxes(automargin=True)
    return fig


def fig_hazard(hz: pd.DataFrame, critics: list[int], titol: str) -> go.Figure:
    """Taxa d'abandonament mensual; marca els mesos crítics (només a la corba global)."""
    fig = go.Figure()
    grups = list(dict.fromkeys(hz["grup"])) if "grup" in hz else [None]
    for i, g in enumerate(grups):
        sub = hz if g is None else hz[hz["grup"] == g]
        nom = "Tots els clients" if g is None else traduir(g)
        fig.add_trace(go.Scatter(
            x=sub["mes"], y=sub["hazard"], mode="lines", name=nom,
            line=dict(color=PALETA[i % len(PALETA)], width=2),
            customdata=np.stack([sub["baixes"], sub["en_risc"]], axis=1),
            hovertemplate=(f"<b>{nom}</b><br>Mes %{{x}}<br>Taxa = %{{y:.2%}}<br>"
                           "%{customdata[0]} baixes de %{customdata[1]} en risc<extra></extra>")))
    if critics and "grup" not in hz:
        crit = hz[hz["mes"].isin(critics)]
        fig.add_trace(go.Scatter(
            x=crit["mes"], y=crit["hazard"], mode="markers", name="Mes crític",
            marker=dict(size=11, color=PALETA[7], symbol="diamond",
                        line=dict(color=SUPERFICIE, width=2)),
            hovertemplate="Mes crític: %{x}<br>Taxa = %{y:.2%}<extra></extra>"))
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    fig.update_xaxes(range=[0.5, 72.5], dtick=6)
    fig.update_layout(showlegend=True)
    return _estil(fig, titol, "Mes d'antiguitat", "Baixes / clients en risc")


def fig_corbes_client(corbes: dict[str, pd.Series], titol: str) -> go.Figure:
    fig = go.Figure()
    for i, (nom, s) in enumerate(corbes.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=nom, line_shape="hv",
            line=dict(color=PALETA[i % len(PALETA)], width=2.5),
            hovertemplate=f"<b>{nom}</b><br>Mes %{{x}}<br>S(t) = %{{y:.1%}}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    return _estil(fig, titol, "Mesos des de l'alta", "Probabilitat de continuar sent client")


def fig_risc_valor(scores: pd.DataFrame, quota: pd.Series,
                   marxat: pd.Series | None = None) -> go.Figure:
    """Matriu risc × valor (valor = quota mensual actual) amb quadrants a les medianes.

    El valor no és el CLV esperat perquè aquest ja incorpora el risc (seria circular).
    `marxat` (booleà per client) separa en una segona sèrie els clients que ja han marxat.
    """
    x = 1 - scores["S_12m"].to_numpy()
    nom_x = "Risc de baixa en 12 mesos (Cox)"
    y = np.asarray(quota, dtype=float)
    mx, my = float(np.median(x)), float(np.median(y))
    custom = np.stack([scores["customerID"], scores["clv_esperat_24m"]], axis=1)
    hover = ("<b>%{customdata[0]}</b><br>Risc = %{x:.1%}<br>Quota = %{y:.2f} €"
             "<br>CLV esperat 24m = %{customdata[1]:,.0f} €<extra>%{fullData.name}</extra>")
    grups = [(np.ones(len(x), dtype=bool), "Clients actius", PALETA[0])]
    if marxat is not None:
        m = np.asarray(marxat, dtype=bool)
        grups = [(~m, "Clients actius", PALETA[0]), (m, "Ja han marxat", PALETA[1])]
    fig = go.Figure()
    for sel, nom, color in grups:
        fig.add_trace(go.Scattergl(
            x=x[sel], y=y[sel], mode="markers", name=nom,
            marker=dict(size=6, color=_rgba(color, 0.45), line=dict(width=0)),
            customdata=custom[sel], hovertemplate=hover, showlegend=marxat is not None))
    fig.add_vline(x=mx, line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"))
    fig.add_hline(y=my, line=dict(color=TEXT_SECUNDARI, width=1, dash="dot"))
    quadrants = [
        (0.98, 0.98, "right", "top", "<b>Retenir primer</b><br>alt risc · alt valor"),
        (0.02, 0.98, "left", "top", "<b>Fidelitzar</b><br>baix risc · alt valor"),
        (0.98, 0.02, "right", "bottom", "<b>Accions de baix cost</b><br>alt risc · baix valor"),
        (0.02, 0.02, "left", "bottom", "<b>Mantenir</b><br>baix risc · baix valor"),
    ]
    for qx, qy, xa, ya, txt in quadrants:
        fig.add_annotation(x=qx, y=qy, xref="paper", yref="paper", text=txt, showarrow=False,
                           xanchor=xa, yanchor=ya, align=xa, font=dict(color=TEXT_PRIMARI, size=12),
                           bgcolor=_rgba(SUPERFICIE, 0.85))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=",.0f", ticksuffix=" €")
    return _estil(fig, "Matriu risc × valor", nom_x, "Quota mensual actual")


def exportar(fig: go.Figure, nom: str) -> list[Path]:
    """Desa sempre l'HTML interactiu; el PNG només si kaleido funciona."""
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    html = DIR_FIGURES / f"{nom}.html"
    fig.write_html(html, include_plotlyjs="directory")  # plotly.min.js local: funciona sense internet
    rutes = [html]
    png = DIR_FIGURES / f"{nom}.png"
    try:
        fig.write_image(png, width=1600, height=900, scale=1)
        rutes.append(png)
    except Exception as e:  # kaleido necessita Chrome; l'HTML ja és el pla B
        print(f"Avís: no s'ha pogut exportar {png.name} ({e}). S'ha desat només l'HTML.")
    return rutes

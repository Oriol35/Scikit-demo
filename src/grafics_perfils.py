"""Gràfics Plotly de la pestanya «Perfils i campanyes» (mateix estil que `src/grafics.py`)."""
import pandas as pd
import plotly.graph_objects as go

from src.grafics import COLOR_PROTECTOR, COLOR_RISC, PALETA, SUPERFICIE, TEXT_PRIMARI, _estil, _rgba


def _pct(x: float) -> str:
    return f"{x:.0%}".replace("%", " %")


def _milers(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def fig_mapa_calor(taxa: pd.DataFrame, n: pd.DataFrame, titol: str) -> go.Figure:
    """Taxa de baixa per contracte × tram d'antiguitat (blau = baixa, taronja = alta)."""
    text = [[f"<b>{_pct(taxa.iloc[i, j])}</b><br><span style='font-size:10px'>n = "
             f"{_milers(n.iloc[i, j])}</span>" for j in range(taxa.shape[1])]
            for i in range(taxa.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=taxa.values, x=[f"{c} mesos" for c in taxa.columns], y=list(taxa.index),
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=TEXT_PRIMARI),
        colorscale=[[0, _rgba(COLOR_PROTECTOR, 0.15)], [0.5, _rgba(COLOR_RISC, 0.35)],
                    [1, _rgba(COLOR_RISC, 0.85)]],
        zmin=0, zmax=0.6, xgap=3, ygap=3,
        colorbar=dict(title="Taxa de<br>baixa", tickformat=".0%", thickness=12),
        hovertemplate="Contracte %{y} · %{x}<br>Han marxat: %{z:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(height=330)
    return _estil(fig, titol, "Antiguitat del client", "Contracte")


def fig_perfils(resum: pd.DataFrame, titol: str) -> go.Figure:
    """Taxa de baixa de cada perfil comparada amb el seu grup de referència."""
    r = resum.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=r["perfil"], x=r["taxa_ref"], orientation="h", name="Grup de referència",
        marker_color=COLOR_PROTECTOR, text=[_pct(v) for v in r["taxa_ref"]], textposition="outside",
        customdata=r["referencia"],
        hovertemplate="<b>Referència:</b> %{customdata}<br>Han marxat: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=r["perfil"], x=r["taxa"], orientation="h", name="Perfil de risc",
        marker_color=COLOR_RISC, text=[_pct(v) for v in r["taxa"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Han marxat: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 0.75], tickformat=".0%")
    fig.update_layout(height=380, barmode="group", bargap=0.3)
    fig = _estil(fig, titol, "Percentatge de clients que han marxat", "")
    fig.update_yaxes(automargin=True)
    return fig


def fig_campanya(resultats: dict[str, dict], titol: str) -> go.Figure:
    """Ingressos salvats, cost del descompte i benefici net de cada escenari (12 mesos)."""
    escenaris = list(resultats)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=escenaris, y=[resultats[e]["ingressos_salvats"] for e in escenaris],
                         name="Ingressos salvats", marker_color=PALETA[0],
                         hovertemplate="%{x}: %{y:,.0f} €<extra>Ingressos salvats</extra>"))
    fig.add_trace(go.Bar(x=escenaris, y=[-resultats[e]["cost_descompte"] for e in escenaris],
                         name="Cost del descompte", marker_color=_rgba(COLOR_RISC, 0.6),
                         hovertemplate="%{x}: %{y:,.0f} €<extra>Cost del descompte</extra>"))
    benefici = [resultats[e]["benefici"] for e in escenaris]
    fig.add_trace(go.Scatter(
        x=escenaris, y=benefici, mode="markers", name="Benefici net",
        marker=dict(size=16, symbol="diamond", color=TEXT_PRIMARI, line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="%{x}: %{y:,.0f} €<extra>Benefici net</extra>"))
    for e, b in zip(escenaris, benefici):
        fig.add_annotation(x=e, y=b, text=f"<b>{'+' if b >= 0 else '−'}{_milers(abs(b))} €</b>",
                           showarrow=False, xshift=58, font=dict(size=13, color=TEXT_PRIMARI),
                           bgcolor=SUPERFICIE, borderpad=3)
    fig.add_hline(y=0, line=dict(color=TEXT_PRIMARI, width=1))
    fig.update_yaxes(tickformat=",.0f", ticksuffix=" €")
    fig.update_layout(height=400, barmode="relative", bargap=0.45, separators=",.")
    fig = _estil(fig, titol, "", "")
    fig.update_yaxes(automargin=True)
    return fig

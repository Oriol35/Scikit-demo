import plotly.graph_objects as go

from src import grafics
from src.grafics import exportar, fig_km
from src.km import ajustar_km
from src.preparacio import carregar_dades


def test_fig_km_una_traca_per_grup():
    f = ajustar_km(carregar_dades(), "Contract")
    fig = fig_km(f, "t", interval=False)
    assert len(fig.data) == 3


def test_exportar_png_falla_html_igualment(tmp_path, monkeypatch):
    monkeypatch.setattr(grafics, "DIR_FIGURES", tmp_path)

    def falla(*a, **k):
        raise RuntimeError("no chrome")

    monkeypatch.setattr(go.Figure, "write_image", falla)
    rutes = exportar(go.Figure(), "prova")
    assert rutes == [tmp_path / "prova.html"] and rutes[0].exists()


def test_esglaons_coordenades_explicites():
    import numpy as np
    from src.grafics import esglaons
    x, y = esglaons(np.array([0, 1, 3]), np.array([1.0, 0.8, 0.5]))
    assert list(x) == [0, 1, 1, 3, 3]
    assert list(y) == [1.0, 1.0, 0.8, 0.8, 0.5]

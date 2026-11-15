from streamlit.testing.v1 import AppTest

from src.preparacio import ARREL

APP = str(ARREL / "app" / "dashboard.py")
PESTANYES = ["Resum", "Kaplan-Meier", "Model de Cox", "Risc temporal",
             "Simulador de client", "Risc × valor"]


def test_app_arrenca_sense_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == PESTANYES


def test_app_recalcula_scores_si_no_hi_ha_fitxer(tmp_path, monkeypatch):
    monkeypatch.setenv("DIR_OUTPUTS", str(tmp_path))  # carpeta buida: sense scores_supervivencia.csv
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("n = 5163" in c.value for c in at.caption)


def test_toggle_estratificat_mostra_ph_del_model_estratificat():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.toggle[1].set_value(True).run()   # toggle[0] = interval KM, toggle[1] = estratificat
    assert not at.exception
    textos = " ".join(c.value for c in at.caption)
    assert "model estratificat" in textos and "quota mensual" in textos


def test_toggle_incloure_baixes_risc_valor(tmp_path, monkeypatch):
    monkeypatch.setenv("DIR_OUTPUTS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert any("n = 5163" in c.value for c in at.caption)
    at.toggle(key="incloure_baixes").set_value(True).run()
    assert not at.exception
    assert any("n = 7032" in c.value for c in at.caption)
    assert any("ja han marxat" in w.value for w in at.warning)


def test_simulador_sense_internet_limita_quota():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key="A_InternetService").set_value("No").run()
    assert not at.exception
    sliders = [s for s in at.slider if s.key.startswith("A_mc")]
    assert len(sliders) == 1 and sliders[0].max <= 30 and sliders[0].value <= 30


def test_barra_lateral_calcula_exclosos():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert any("11 registres" in c.value for c in at.sidebar.caption)


def test_textos_explicatius_valors_p():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    tot = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "significativament diferents" in tot          # conclusió log-rank
    assert "Interpretació" in " ".join(str(d.value.columns.tolist()) for d in at.dataframe)
    assert "72 mesos" in tot                               # límit temporal explicat
    assert "test" in tot and "no ha vist" in tot           # explicació del C-index

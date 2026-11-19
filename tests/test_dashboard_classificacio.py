from streamlit.testing.v1 import AppTest

from src.preparacio import ARREL

APP = str(ARREL / "app" / "dashboard.py")


def _app():
    return AppTest.from_file(APP, default_timeout=300).run()


def test_pestanya_classificacio_amb_ajudes():
    at = _app()
    assert not at.exception
    ajudes = {m.label: m.proto.help for m in at.metric}
    for metrica in ["AUC", "Recall", "Precision", "F1", "Exactitud", "Lift"]:
        assert ajudes.get(metrica), f"Falta l'explicació de la mètrica {metrica}"
    assert any("Què és:" in i.value for i in at.info)               # explicació del model


def test_canviar_de_model_i_tractament():
    at = _app()
    for nom in ["Arbre de decisió", "Random Forest", "Gradient boosting"]:
        at.selectbox(key="cl_model").set_value(nom).run()
        assert not at.exception
        assert any(nom in i.value or "Què és:" in i.value for i in at.info)
    at.selectbox(key="cl_tractament").set_value("SMOTE").run()
    assert not at.exception
    assert any("SMOTE" in c.value for c in at.caption)


def test_simulador_classificacio_mostra_cami_arbre():
    at = _app()
    at.selectbox(key="cl_model").set_value("Arbre de decisió").run()
    at.selectbox(key="cl_InternetService").set_value("No").run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Resultat" in md and "probabilitat de baixa" in md

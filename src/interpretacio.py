"""Traducció dels resultats estadístics a frases en català (per al dashboard i l'informe).

Nivell de significació: 5 % (alfa = 0,05).
"""
ALFA = 0.05


def _p(p: float) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def conclusio_logrank(p: float, variable: str, millor: str | None = None,
                      pitjor: str | None = None) -> str:
    """Conclusió del test log-rank (H0: totes les corbes de supervivència són iguals)."""
    if p < ALFA:
        t = (f"**Conclusió:** les corbes són **significativament diferents** ({_p(p)}). "
             f"La variable «{variable}» influeix en la permanència dels clients: si les corbes fossin "
             "iguals, una diferència com aquesta passaria per atzar menys d'un 5 % de les vegades.")
        if millor and pitjor:
            t += f" El grup que més es queda és **{millor}** i el que més marxa, **{pitjor}**."
        return t
    return (f"**Conclusió:** no hi ha prou evidència per dir que les corbes siguin diferents "
            f"({_p(p)} ≥ 0,05). Les diferències que es veuen poden ser fruit de l'atzar.")


def interpretar_hr(hr: float, p: float) -> str:
    """Lectura d'un hazard ratio: canvi percentual del risc instantani de baixa."""
    if p >= ALFA:
        return f"Sense efecte demostrat ({_p(p)})"
    if hr < 1:
        return f"Redueix el risc un {round((1 - hr) * 100)} % (significatiu)"
    return f"Augmenta el risc un {round((hr - 1) * 100)} % (significatiu)"


def interpretar_ph(p: float) -> str:
    """Lectura del test de riscos proporcionals (H0: l'efecte és constant en el temps)."""
    if p < ALFA:
        return "L'efecte canvia amb el temps (viola el supòsit)"
    return "Efecte constant en el temps (compleix el supòsit)"


def text_mediana(mediana: str) -> str:
    """La mediana no assolida vol dir que S(t) no baixa del 50 % dins l'horitzó observat."""
    return "> 72 (no s'assoleix)" if mediana == "no assolida" else mediana

"""Models de classificació del churn: «marxarà aquest client, sí o no?».

Quatre models (regressió logística, arbre de decisió, Random Forest i gradient boosting), tres
tractaments del desequilibri de classes (cap, pesos de classe i SMOTE), mètriques d'avaluació,
lift per a campanyes de retenció i eines d'interpretació (odds ratios, regles de l'arbre i
importància de les variables).

Es fan servir la mateixa neteja (`carregar_dades`) i la mateixa partició train/test
(`dividir_train_test`) que l'anàlisi de supervivència, perquè els resultats siguin comparables.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.preparacio import REFERENCIES, dividir_train_test

LLAVOR = 42  # random_state comú: resultats reproduïbles

# ---------- Variables ----------
VARIABLES_NUMERIQUES = ["tenure", "MonthlyCharges"]
VARIABLES_CATEGORIQUES = [
    "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies", "PhoneService",
    "MultipleLines", "PaperlessBilling", "Partner", "Dependents", "gender",
]
# A diferència del model de Cox, aquí `tenure` és una variable predictiva més (al Cox és el temps).
# `TotalCharges` s'exclou: és pràcticament tenure × MonthlyCharges (informació repetida).
VARIABLES_MODEL = VARIABLES_NUMERIQUES + ["SeniorCitizen"] + VARIABLES_CATEGORIQUES

# Categoria de referència (la que no té columna pròpia). Les comunes són les mateixes que les
# del model de Cox, perquè els odds ratios i els hazard ratios es llegeixin igual.
REFERENCIES_CLASSIFICACIO = {
    **REFERENCIES,
    "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "PhoneService": "No", "MultipleLines": "No", "gender": "Female",
}

# Valors «sense servei» que no aporten informació nova (ja la dona InternetService/PhoneService).
SERVEIS_INTERNET = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection",
                    "StreamingTV", "StreamingMovies"]

NOMS_VARIABLES = {
    "tenure": "Antiguitat", "MonthlyCharges": "Quota mensual", "SeniorCitizen": "Client sènior",
    "Contract": "Tipus de contracte", "InternetService": "Servei d'internet",
    "PaymentMethod": "Mètode de pagament", "OnlineSecurity": "Seguretat en línia",
    "TechSupport": "Suport tècnic", "OnlineBackup": "Còpia de seguretat en línia",
    "DeviceProtection": "Protecció de dispositius", "StreamingTV": "Televisió en streaming",
    "StreamingMovies": "Pel·lícules en streaming", "PhoneService": "Telefonia",
    "MultipleLines": "Diverses línies", "PaperlessBilling": "Factura electrònica",
    "Partner": "Té parella", "Dependents": "Té persones a càrrec", "gender": "Gènere",
}

NOMS_COLUMNES = {
    "tenure": "Antiguitat (+1 mes)",
    "MonthlyCharges": "Quota mensual (+1 €)",
    "SeniorCitizen": "Client sènior",
    "Contract_One year": "Contracte 1 any (vs mensual)",
    "Contract_Two year": "Contracte 2 anys (vs mensual)",
    "InternetService_Fiber optic": "Fibra òptica (vs DSL)",
    "InternetService_No": "Sense internet (vs DSL)",
    "PaymentMethod_Bank transfer (automatic)": "Transferència automàtica (vs xec postal)",
    "PaymentMethod_Credit card (automatic)": "Targeta automàtica (vs xec postal)",
    "PaymentMethod_Electronic check": "Xec electrònic (vs xec postal)",
    "OnlineSecurity_Yes": "Seguretat en línia",
    "TechSupport_Yes": "Suport tècnic",
    "OnlineBackup_Yes": "Còpia de seguretat en línia",
    "DeviceProtection_Yes": "Protecció de dispositius",
    "StreamingTV_Yes": "Televisió en streaming",
    "StreamingMovies_Yes": "Pel·lícules en streaming",
    "PhoneService_Yes": "Té telefonia",
    "MultipleLines_Yes": "Diverses línies",
    "PaperlessBilling_Yes": "Factura electrònica",
    "Partner_Yes": "Té parella",
    "Dependents_Yes": "Té persones a càrrec",
    "gender_Male": "Home (vs dona)",
}

# Pregunta curta per als nodes de l'arbre (columnes 0/1).
PREGUNTES_ARBRE = {
    "Contract_One year": "Contracte d'1 any?", "Contract_Two year": "Contracte de 2 anys?",
    "InternetService_Fiber optic": "Té fibra òptica?", "InternetService_No": "No té internet?",
    "PaymentMethod_Bank transfer (automatic)": "Paga per transferència?",
    "PaymentMethod_Credit card (automatic)": "Paga amb targeta?",
    "PaymentMethod_Electronic check": "Paga amb xec electrònic?",
    "OnlineSecurity_Yes": "Té seguretat en línia?", "TechSupport_Yes": "Té suport tècnic?",
    "OnlineBackup_Yes": "Té còpia en línia?", "DeviceProtection_Yes": "Té protecció de dispositius?",
    "StreamingTV_Yes": "Té TV en streaming?", "StreamingMovies_Yes": "Té pel·lícules en streaming?",
    "PhoneService_Yes": "Té telefonia?", "MultipleLines_Yes": "Té diverses línies?",
    "PaperlessBilling_Yes": "Factura electrònica?", "Partner_Yes": "Té parella?",
    "Dependents_Yes": "Té persones a càrrec?", "gender_Male": "És home?",
    "SeniorCitizen": "És client sènior?",
}
UNITATS = {"tenure": "mesos", "MonthlyCharges": "€"}

# ---------- Models i tractaments ----------
MODELS = ["Regressió logística", "Arbre de decisió", "Random Forest", "Gradient boosting"]
TRACTAMENTS = ["Cap", "Pesos de classe", "SMOTE"]


def _classificador(nom: str, pesos: bool):
    cw = "balanced" if pesos else None
    if nom == "Regressió logística":
        return LogisticRegression(max_iter=5000, class_weight=cw)
    if nom == "Arbre de decisió":
        # Profunditat 4 i fulles de ≥ 50 clients: prou precís i encara llegible com a regles.
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight=cw,
                                      random_state=LLAVOR)
    if nom == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight=cw,
                                      random_state=LLAVOR, n_jobs=-1)
    if nom == "Gradient boosting":
        # Implementació de scikit-learn (equivalent a XGBoost/LightGBM per a aquest volum).
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_depth=3,
                                              class_weight=cw, random_state=LLAVOR)
    raise ValueError(f"Model desconegut: {nom}")


# ---------- Preparació ----------
def _normalitzar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SERVEIS_INTERNET:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def codificar_classificacio(df: pd.DataFrame, columnes: list[str] | None = None) -> pd.DataFrame:
    """Matriu de variables (X): numèriques primer i després columnes 0/1 (one-hot sense referència).

    Si es donen `columnes`, el resultat té exactament aquestes columnes (per a perfils simulats).
    """
    df = _normalitzar(df)
    parts = [df[VARIABLES_NUMERIQUES].astype(float), df[["SeniorCitizen"]].astype(int)]
    for var in VARIABLES_CATEGORIQUES:
        d = pd.get_dummies(df[var], prefix=var, prefix_sep="_", dtype=int)
        d = d.drop(columns=f"{var}_{REFERENCIES_CLASSIFICACIO[var]}", errors="ignore")
        parts.append(d)
    X = pd.concat(parts, axis=1)
    if columnes is not None:
        X = X.reindex(columns=columnes, fill_value=0)
    return X


def codificar_perfil_classificacio(perfil: dict, columnes: list[str]) -> pd.DataFrame:
    """Codifica un client simulat (valors originals del dataset) amb les columnes del model."""
    perfil = dict(perfil)
    if perfil.get("InternetService") == "No":
        for col in SERVEIS_INTERNET:
            perfil[col] = "No"
    if perfil.get("PhoneService") == "No":
        perfil["MultipleLines"] = "No"
    return codificar_classificacio(pd.DataFrame([perfil]), columnes)


def preparar_dades(df: pd.DataFrame) -> dict:
    """X i y d'entrenament (80 %) i de prova (20 %), amb la partició de l'anàlisi de supervivència."""
    tr, te = dividir_train_test(df)
    X_tr = codificar_classificacio(tr)
    X_te = codificar_classificacio(te, list(X_tr.columns))
    return {"X_tr": X_tr, "y_tr": tr["event"].to_numpy(), "X_te": X_te,
            "y_te": te["event"].to_numpy(), "train": tr, "test": te,
            "columnes": list(X_tr.columns)}


def ajustar_model(nom: str, tractament: str, X_tr: pd.DataFrame, y_tr: np.ndarray) -> Pipeline:
    """Pipeline: escalat de les numèriques → (SMOTE) → classificador.

    SMOTE va dins del pipeline: només s'aplica en entrenar (`fit`), mai a les dades de prova.
    S'usa SMOTENC, la variant de SMOTE que respecta les variables 0/1 (no inventa «mig contracte»).
    """
    if tractament not in TRACTAMENTS:
        raise ValueError(f"Tractament desconegut: {tractament}")
    n_num = len(VARIABLES_NUMERIQUES)
    escalat = ColumnTransformer([("num", StandardScaler(), list(range(n_num)))],
                                remainder="passthrough")
    model = ("model", _classificador(nom, pesos=tractament == "Pesos de classe"))
    if tractament == "SMOTE":
        # Importació diferida: imbalanced-learn només es carrega si es tria SMOTE (vegeu
        # `smote_disponible`). El seu Pipeline és com el de scikit-learn però admet mostrejadors.
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import Pipeline as PipelineMostreig
        smote = SMOTENC(categorical_features=list(range(n_num, X_tr.shape[1])), random_state=LLAVOR)
        pipeline = PipelineMostreig([("escalat", escalat), ("smote", smote), model])
    else:
        pipeline = Pipeline([("escalat", escalat), model])
    return pipeline.fit(X_tr.to_numpy(dtype=float), y_tr)


def smote_disponible() -> bool:
    """Comprova si imbalanced-learn (SMOTE) es pot carregar en aquest ordinador.

    En alguns Windows 11, el Control intel·ligent d'aplicacions bloqueja una DLL de scikit-learn
    que imbalanced-learn carrega en importar-se. Així, la resta de models funcionen igualment.
    """
    try:
        import imblearn  # noqa: F401
        return True
    except Exception:  # ImportError o error de càrrega de DLL
        return False


def probabilitats(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilitat de baixa (classe 1) per a cada client."""
    return model.predict_proba(X.to_numpy(dtype=float))[:, 1]


def ajustar_tots(tractament: str, dades: dict) -> dict[str, Pipeline]:
    return {nom: ajustar_model(nom, tractament, dades["X_tr"], dades["y_tr"]) for nom in MODELS}


# ---------- Mètriques ----------
def avaluar(y: np.ndarray, p: np.ndarray, llindar: float = 0.5) -> dict:
    """Mètriques al conjunt de prova. La decisió és «en risc» si la probabilitat ≥ llindar."""
    pred = (p >= llindar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "exactitud": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)),
        "vp": int(vp), "fp": int(fp), "fn": int(fn), "vn": int(vn),
    }


def corba_roc(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    fpr, tpr, llindars = roc_curve(y, p)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "llindar": llindars})


def corba_guanys(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Clients ordenats de més a menys risc: % contactat → % de baixes capturades i lift."""
    ordre = np.argsort(-p, kind="stable")
    capturades = np.cumsum(np.asarray(y)[ordre])
    n = len(y)
    pct_contactat = np.arange(1, n + 1) / n
    pct_capturat = capturades / max(capturades[-1], 1)
    return pd.DataFrame({"pct_contactat": np.r_[0, pct_contactat],
                         "pct_capturat": np.r_[0, pct_capturat],
                         "lift": np.r_[np.nan, pct_capturat / pct_contactat]})


def campanya(y: np.ndarray, p: np.ndarray, pct: float) -> dict:
    """Resultat de contactar el `pct` (0–1) de clients amb més risc segons el model."""
    n = len(y)
    k = max(1, int(round(pct * n)))
    top = np.argsort(-p, kind="stable")[:k]
    detectades = int(np.asarray(y)[top].sum())
    totals = int(np.sum(y))
    taxa_base = totals / n
    return {"contactats": k, "baixes_detectades": detectades, "baixes_totals": totals,
            "pct_capturat": detectades / totals if totals else 0.0,
            "precisio": detectades / k, "taxa_base": taxa_base,
            "lift": (detectades / k) / taxa_base if taxa_base else float("nan")}


def taula_comparativa(models: dict[str, Pipeline], dades: dict, llindar: float = 0.5,
                      pct_campanya: float = 0.1) -> pd.DataFrame:
    files = []
    for nom, model in models.items():
        p = probabilitats(model, dades["X_te"])
        m = avaluar(dades["y_te"], p, llindar)
        c = campanya(dades["y_te"], p, pct_campanya)
        files.append({"model": nom, "auc": m["auc"], "recall": m["recall"],
                      "precision": m["precision"], "f1": m["f1"], "exactitud": m["exactitud"],
                      "lift": c["lift"]})
    return pd.DataFrame(files)


# ---------- Interpretació ----------
def _escalat(model: Pipeline) -> StandardScaler:
    return model.named_steps["escalat"].named_transformers_["num"]


def odds_ratios(model: Pipeline, columnes: list[str]) -> pd.DataFrame:
    """Odds ratio de cada columna de la regressió logística, en unitats originals.

    Les numèriques s'escalen abans d'entrenar; dividint el coeficient per la desviació típica
    es torna a «per 1 mes» o «per 1 €».
    """
    clf = model.named_steps["model"]
    if not isinstance(clf, LogisticRegression):
        raise TypeError("Els odds ratios només existeixen per a la regressió logística")
    coef = clf.coef_[0].copy()
    n_num = len(VARIABLES_NUMERIQUES)
    coef[:n_num] = coef[:n_num] / _escalat(model).scale_
    t = pd.DataFrame({"variable": columnes, "coef": coef, "OR": np.exp(coef)})
    t["nom"] = t["variable"].map(NOMS_COLUMNES).fillna(t["variable"])
    return t.sort_values("OR").reset_index(drop=True)


def _grups(columnes: list[str]) -> dict[str, list[str]]:
    """Columnes codificades agrupades per variable original (Contract → 2 columnes, etc.)."""
    grups = {}
    for var in VARIABLES_MODEL:
        cols = [c for c in columnes if c == var or c.startswith(f"{var}_")]
        if cols:
            grups[var] = cols
    return grups


def importancia_variables(model: Pipeline, X: pd.DataFrame, y: np.ndarray,
                          repeticions: int = 5) -> pd.DataFrame:
    """Importància per permutació, agrupada per variable original i mesurada amb l'AUC.

    Es barregen a l'atzar els valors d'una variable (totes les seves columnes alhora): si l'AUC
    cau molt, el model depenia d'aquella variable. Funciona igual per als quatre models.
    """
    rng = np.random.default_rng(LLAVOR)
    base = roc_auc_score(y, probabilitats(model, X))
    files = []
    for var, cols in _grups(list(X.columns)).items():
        caigudes = []
        for _ in range(repeticions):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            caigudes.append(base - roc_auc_score(y, probabilitats(model, Xp)))
        files.append({"variable": var, "nom": NOMS_VARIABLES[var],
                      "importancia": float(np.mean(caigudes)), "desviacio": float(np.std(caigudes))})
    return pd.DataFrame(files).sort_values("importancia", ascending=False).reset_index(drop=True)


def _num(v: float) -> str:
    return f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".").replace(",0", "")


def estructura_arbre(model: Pipeline, columnes: list[str], X_tr: pd.DataFrame,
                     y_tr: np.ndarray) -> pd.DataFrame:
    """Nodes de l'arbre en llenguatge planer, amb els llindars en unitats originals.

    `n` i `taxa_real` es calculen amb els clients reals d'entrenament que arriben a cada node
    (sense pesos ni clients sintètics); `prob_model` és la probabilitat que dona el model.
    """
    clf = model.named_steps["model"]
    if not isinstance(clf, DecisionTreeClassifier):
        raise TypeError("Només per al model «Arbre de decisió»")
    arbre = clf.tree_
    sc = _escalat(model)
    Xt = model.named_steps["escalat"].transform(X_tr.to_numpy(dtype=float))
    camins = clf.decision_path(Xt).toarray().astype(bool)
    y_tr = np.asarray(y_tr)
    pares, costat = {}, {}
    for i in range(arbre.node_count):
        for fill, c in [(arbre.children_left[i], "esquerra"), (arbre.children_right[i], "dreta")]:
            if fill != -1:
                pares[fill], costat[fill] = i, c
    files = []
    for i in range(arbre.node_count):
        fulla = arbre.children_left[i] == -1
        valors = arbre.value[i][0]
        prob = float(valors[1] / valors.sum())
        dins = camins[:, i]
        pregunta = si = no = ""
        if not fulla:
            j = arbre.feature[i]
            col = columnes[j]
            llindar = arbre.threshold[i]
            if j < len(VARIABLES_NUMERIQUES):
                valor = llindar * sc.scale_[j] + sc.mean_[j]
                pregunta = f"{NOMS_VARIABLES[col]} ≤ {_num(valor)} {UNITATS[col]}?"
                si, no = "esquerra", "dreta"   # x ≤ llindar → branca esquerra
            else:
                pregunta = PREGUNTES_ARBRE.get(col, f"{col}?")
                si, no = "dreta", "esquerra"   # columna 0/1: > 0,5 vol dir «sí»
        files.append({
            "node": i, "pare": pares.get(i, -1), "costat": costat.get(i, ""),
            "fulla": bool(fulla), "pregunta": pregunta, "branca_si": si, "branca_no": no,
            "n": int(dins.sum()), "taxa_real": float(y_tr[dins].mean()) if dins.any() else np.nan,
            "prob_model": prob,
        })
    t = pd.DataFrame(files)
    prof = {0: 0}
    for i in range(1, arbre.node_count):
        prof[i] = prof[pares[i]] + 1
    t["profunditat"] = t["node"].map(prof)
    return t


def nodes_cami(model: Pipeline, fila: pd.DataFrame) -> list[int]:
    """Nodes de l'arbre pels quals passa un client (de l'arrel a la fulla)."""
    clf = model.named_steps["model"]
    Xt = model.named_steps["escalat"].transform(fila.to_numpy(dtype=float))
    return [int(n) for n in clf.decision_path(Xt).indices]


def cami_arbre(model: Pipeline, fila: pd.DataFrame, estructura: pd.DataFrame) -> list[dict]:
    """Preguntes que fa l'arbre per a un client concret i la resposta a cadascuna."""
    nodes = nodes_cami(model, fila)
    est = estructura.set_index("node")
    passos = []
    for actual, seguent in zip(nodes[:-1], nodes[1:]):
        fila_est = est.loc[actual]
        costat = est.loc[seguent, "costat"]
        passos.append({"pregunta": fila_est["pregunta"],
                       "resposta": "Sí" if costat == fila_est["branca_si"] else "No"})
    final = est.loc[nodes[-1]]
    passos.append({"pregunta": "Resultat", "resposta":
                   f"probabilitat de baixa {final['prob_model']:.0%} "
                   f"({final['n']} clients reals en aquesta fulla, "
                   f"{final['taxa_real']:.0%} van marxar)"})
    return passos


# ---------- Textos explicatius (dashboard i notebook) ----------
DESCRIPCIONS_MODELS = {
    "Regressió logística": (
        "**Què és:** suma les variables del client, cadascuna multiplicada per un pes, i converteix "
        "el resultat en una probabilitat entre 0 i 1 (funció sigmoide, en forma de «S»).  \n"
        "**Punt fort:** és el més fàcil d'explicar: cada variable té un **odds ratio** que diu quant "
        "multiplica la probabilitat relativa de marxar.  \n"
        "**Punt feble:** suposa que cada variable té un efecte fix i independent de les altres, i "
        "no capta combinacions complexes."),
    "Arbre de decisió": (
        "**Què és:** una seqüència de preguntes sí/no (com el joc «Endevina qui?»). El model ha "
        "après sol quines preguntes fer, en quin ordre i on tallar (p. ex. «antiguitat ≤ 16 mesos?»).  \n"
        "**Punt fort:** es llegeix com a **regles de negoci** i es pot seguir el camí de cada client.  \n"
        "**Punt feble:** si creix massa s'aprèn les dades de memòria (sobreajust). Aquí es limita a "
        "4 nivells i a fulles d'almenys 50 clients."),
    "Random Forest": (
        "**Què és:** 300 arbres diferents, cadascun entrenat amb una mostra a l'atzar de clients i "
        "de variables. La probabilitat final és la **mitjana dels vots** de tots els arbres.  \n"
        "**Punt fort:** robust i precís sense gaire ajust; corregeix la inestabilitat d'un sol arbre.  \n"
        "**Punt feble:** no hi ha cap regla única per llegir: s'explica amb la **importància de les "
        "variables**."),
    "Gradient boosting": (
        "**Què és:** arbres petits que s'entrenen **un darrere l'altre**: cada arbre nou es "
        "concentra a corregir els errors dels anteriors (com un equip que es passa l'examen per "
        "torns).  \n"
        "**Punt fort:** sol ser el més precís amb dades de clients (taules).  \n"
        "**Punt feble:** és el menys interpretable (una «caixa negra»): s'explica amb la "
        "**importància de les variables**. Implementació de scikit-learn, equivalent a XGBoost."),
}

DESCRIPCIONS_TRACTAMENTS = {
    "Cap": "S'entrena amb les dades tal com són (≈ 27 % de baixes). El model tendeix a dir "
           "«no marxa» i se li escapen baixes.",
    "Pesos de classe": "Els errors amb els clients que marxen **compten més** en entrenar (en "
                       "proporció inversa a la seva freqüència). No es modifiquen les dades.",
    "SMOTE": "Es creen **clients sintètics** que marxen, a mig camí entre dos clients reals "
             "semblants, fins a equilibrar les classes. Només a l'entrenament, mai a la prova.",
}

AJUDA_METRIQUES = {
    "auc": "Probabilitat que, triant a l'atzar un client que marxa i un que no, el model doni més "
           "risc al que marxa. 0,5 = a l'atzar; 1 = perfecte. No depèn del llindar.",
    "recall": "Dels clients que van marxar de veritat, quin percentatge ha detectat el model. "
              "És la mètrica clau en churn: un client no detectat és un client perdut.",
    "precision": "Dels clients que el model assenyala com a «en risc», quin percentatge marxen de "
                 "veritat. Si és baixa, es fan moltes ofertes a clients que no marxaven.",
    "f1": "Mitjana harmònica de precision i recall: un sol número que resumeix l'equilibri entre "
          "detectar baixes i no fer massa falses alarmes.",
    "exactitud": "Percentatge total d'encerts. Amb classes desequilibrades enganya: un model que "
                 "digués sempre «no marxa» encertaria un 73 % sense detectar cap baixa.",
    "lift": "Quantes vegades més baixes es troben contactant els clients amb més risc segons el "
            "model que no pas triant clients a l'atzar. Lift 3 = el triple.",
    "llindar": "A partir de quina probabilitat es considera que un client està «en risc». Baixar-lo "
               "detecta més baixes (més recall) a canvi de més falses alarmes (menys precision).",
}


def frase_metriques(m: dict) -> str:
    """Lectura en llenguatge planer de la matriu de confusió."""
    marxen = m["vp"] + m["fn"]
    alertes = m["vp"] + m["fp"]
    return (f"Dels **{marxen}** clients de prova que van marxar, el model n'ha detectat **{m['vp']}** "
            f"(recall {m['recall']:.0%}) i se n'han escapat {m['fn']}. Dels **{alertes}** clients que "
            f"ha marcat com a «en risc», **{m['vp']}** van marxar de veritat (precision "
            f"{m['precision']:.0%}); els altres {m['fp']} són falses alarmes.")


def interpretar_or(or_: float) -> str:
    """Lectura d'un odds ratio (efecte mantenint la resta de variables iguals)."""
    if abs(or_ - 1) < 0.02:
        return "Pràcticament sense efecte"
    if or_ < 1:
        return f"Redueix les odds de baixa un {round((1 - or_) * 100)} %"
    return f"Multiplica les odds de baixa per {or_:.2f}".replace(".", ",")

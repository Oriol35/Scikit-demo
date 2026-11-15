# Cas pràctic — Anàlisi de supervivència i retenció (Telco Customer Churn)

## Estructura

```
data/        dataset original
docs/        informe metodològic i guió de la demo
src/         codi comú de notebooks i dashboard (preparació, Kaplan-Meier, Cox, risc temporal, gràfics)
notebooks/   01 preparació · 02 Kaplan-Meier · 03 Cox · 04 risc temporal i CLV
app/         dashboard Streamlit per a la demo
figures/     gràfics exportats (HTML interactiu sense internet + PNG per a l'informe)
outputs/     scores_supervivencia.csv (risc, supervivència i CLV per client)
tests/       proves automàtiques (pytest)
```

## Posar-ho en marxa en qualsevol ordinador

Requisit: **Python 3.11 o superior** (https://www.python.org/downloads/). La carpeta `.venv` no es
copia entre ordinadors: es crea de nou amb l'script d'instal·lació (cal internet només aquest cop).

| | Windows | macOS / Linux |
|---|---|---|
| 1. Instal·lar (un sol cop) | doble clic a `instal-lar.bat` | `./instal-lar.sh` |
| 2. Obrir el dashboard | doble clic a `demo.bat` | `./demo.sh` |

El dashboard s'obre a http://localhost:8501 i funciona **sense connexió** un cop instal·lat.

**Pla B sense Python:** obrir els fitxers `figures/*.html` amb qualsevol navegador (interactius,
sense internet).

## Altres ordres (des d'aquesta carpeta, Windows)

```bash
.venv\Scripts\python.exe -m pytest
cd notebooks && ..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Els notebooks també es poden obrir a VS Code (kernel `.venv`).

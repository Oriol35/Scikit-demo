#!/usr/bin/env bash
# Obre el dashboard al navegador (http://localhost:8501).
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Executa primer ./instal-lar.sh"; exit 1; }
.venv/bin/python -m streamlit run app/dashboard.py

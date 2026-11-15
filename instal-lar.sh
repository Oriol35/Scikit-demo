#!/usr/bin/env bash
# Crea l'entorn virtual i instal·la les dependències (macOS / Linux; cal Python 3.11+).
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
echo "Instal·lació completada. Executa ./demo.sh per obrir el dashboard."

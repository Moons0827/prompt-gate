#!/usr/bin/env bash
set -e
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp env.example .env
python seed.py
uvicorn app.main:app --reload --port 8000

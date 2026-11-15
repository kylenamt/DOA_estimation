#!/usr/bin/env bash
# Train the DOA model. Any arguments are passed to train.py, e.g.
#   ./train.sh --epochs 50 --batch-size 256 --half-fov 25 --save models/fov25.pth
# The output is also written to logs/<timestamp>.log
set -e
cd "$(dirname "$0")"

# Python from the project venv (Windows or Linux/macOS layout)
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S).log"
"$PY" -u train.py "$@" 2>&1 | tee "$LOG"

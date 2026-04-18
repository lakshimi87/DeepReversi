#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/venvs/torch/bin/activate"
python "$SCRIPT_DIR/ReversiRL/reversiRL.py" test-random --games 100 "$@"

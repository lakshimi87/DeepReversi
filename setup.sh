#!/bin/bash
set -e

VENV_DIR="$HOME/venvs/torch"

if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at $VENV_DIR"
    read -p "Recreate? [y/N] " ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        rm -rf "$VENV_DIR"
    else
        echo "Skipping venv creation. Installing packages into existing venv..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install torch numpy pygame-ce
        echo "Done!"
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_DIR ..."
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch + numpy..."
pip install torch numpy

echo "Installing pygame-ce..."
pip install pygame-ce

echo ""
echo "Setup complete!"
echo "  venv: $VENV_DIR"
echo "  Activate: source $VENV_DIR/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

_fail_setup() {
    return 1 2>/dev/null || exit 1
}

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    _fail_setup
fi

if [[ "$(uname)" == "Darwin" ]]; then
    REQUIREMENTS="$MODEL_ROOT/requirements-mac.txt"
else
    REQUIREMENTS="$MODEL_ROOT/requirements-linux.txt"
fi

INSTALL_FLAG="$MODEL_ROOT/.installed_$(uname)"

if ! conda env list | grep -q "^rynnvla002 "; then
    echo "Creating conda environment rynnvla002..."
    conda create -n rynnvla002 python=3.13 -y
    rm -f "$INSTALL_FLAG"
fi

if [ ! -f "$INSTALL_FLAG" ]; then
    echo "Installing requirements..."
    conda run -n rynnvla002 pip install -r "$REQUIREMENTS" && touch "$INSTALL_FLAG"
else
    echo "Requirements already installed. Delete $INSTALL_FLAG to force reinstall."
fi

conda activate rynnvla002
export PYTHON=$(conda run -n rynnvla002 which python3)

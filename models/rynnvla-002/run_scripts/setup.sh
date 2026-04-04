export MODEL_ROOT=models/rynnvla-002  # structural — always this path, not configurable

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    return 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    REQUIREMENTS="$MODEL_ROOT/requirements-mac.txt"
else
    REQUIREMENTS="$MODEL_ROOT/requirements-linux.txt"
fi

if ! conda env list | grep -q "^rynnvla002 "; then
    echo "Creating conda environment rynnvla002..."
    conda create -n rynnvla002 python=3.13 -y
    conda run -n rynnvla002 pip install -r "$REQUIREMENTS"
fi

conda activate rynnvla002
export PYTHON=$(conda run -n rynnvla002 which python3)

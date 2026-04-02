if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    return 1
fi

$PYTHON $MODEL_ROOT/preprocessing/convert_lerobot.py

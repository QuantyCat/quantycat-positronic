if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    return 1
fi

echo "--- Step 1: Convert LeRobot dataset to RynnVLA-002 format ---"
$PYTHON $MODEL_ROOT/preprocessing/convert_lerobot.py || return 1

echo "--- Step 2: Generate conversation JSON ---"
$PYTHON $MODEL_ROOT/preprocessing/generate_conversations.py || return 1

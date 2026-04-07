if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True 

$PYTHON $MODEL_ROOT/fine_tuning/finetune.py

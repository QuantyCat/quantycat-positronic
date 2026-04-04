if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    return 1
fi

echo "--- Step 1: Convert LeRobot dataset to RynnVLA-002 format ---"
$PYTHON $MODEL_ROOT/preprocessing/convert_lerobot.py || return 1

echo ""
echo "--- Step 2: Generate conversation JSON ---"
$PYTHON $MODEL_ROOT/preprocessing/generate_conversations.py || return 1

echo ""
echo "--- Step 3: Calculate action and state min/max values ---"
echo "(Copy these into RynnVLA-002/rynnvla-002/data_lerobot/item_processor.py before pretokenizing)"
TRAINING_DATA=$(python3 -c "import yaml; c=yaml.safe_load(open('$MODEL_ROOT/config.yaml')); print(c['work_dir'] + '/training_data')")
$PYTHON $MODEL_ROOT/preprocessing/calculate_min_max_action.py "$TRAINING_DATA" || return 1
$PYTHON $MODEL_ROOT/preprocessing/calculate_min_max_state.py "$TRAINING_DATA" || return 1

echo ""
echo "--- Step 4: Verify outputs ---"
$PYTHON $MODEL_ROOT/preprocessing/verify.py || return 1
echo ""

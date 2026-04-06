if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    return 1
fi

echo "--- Step 1: Convert LeRobot dataset to RynnVLA-002 format ---"
$PYTHON $MODEL_ROOT/preprocessing/step1_convert_lerobot.py || return 1

echo ""
echo "--- Step 2: Generate conversation JSON ---"
$PYTHON $MODEL_ROOT/preprocessing/step2_generate_conversations.py || return 1

echo ""
echo "--- Step 3: Verify outputs ---"
$PYTHON $MODEL_ROOT/preprocessing/step3_verify.py || return 1

echo ""
echo "--- Step 4: Calculate action and state min/max values ---"
$PYTHON $MODEL_ROOT/preprocessing/step4_calculate_min_max.py || return 1

echo ""
echo "--- Step 5: Pretokenize ---"
$PYTHON $MODEL_ROOT/preprocessing/step5_pretokenize.py || return 1

echo ""
echo "--- Step 6: Merge worker records ---"
$PYTHON $MODEL_ROOT/preprocessing/step6_merge_records.py || return 1

echo ""
echo "--- Step 7: Update training config ---"
$PYTHON $MODEL_ROOT/preprocessing/step7_update_train_config.py || return 1

echo ""
echo "Preprocessing complete."

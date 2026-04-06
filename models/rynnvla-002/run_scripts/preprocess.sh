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
echo "--- Step 3: Verify outputs ---"
$PYTHON $MODEL_ROOT/preprocessing/verify.py || return 1

echo ""
echo "--- Step 4: Calculate action and state min/max values ---"
echo "(Copy these into RynnVLA-002/rynnvla-002/data_lerobot/item_processor.py before pretokenizing)"
TRAINING_DATA=$($PYTHON -c "import yaml; c=yaml.safe_load(open('$MODEL_ROOT/config.yaml')); print(c['work_dir'] + '/training_data')")
WORK_DIR=$($PYTHON -c "import yaml; c=yaml.safe_load(open('$MODEL_ROOT/config.yaml')); print(c['work_dir'])")
ACTION_STATS="$WORK_DIR/min_max_action.txt"
STATE_STATS="$WORK_DIR/min_max_state.txt"
if [ -f "$ACTION_STATS" ] && [ -f "$STATE_STATS" ]; then
    echo "Min/max already calculated — skipping. Delete $WORK_DIR/min_max_*.txt to rerun."
    echo "Action stats: $ACTION_STATS"
    echo "State stats:  $STATE_STATS"
else
    $PYTHON $MODEL_ROOT/preprocessing/calculate_min_max_action.py "$TRAINING_DATA" | tee "$ACTION_STATS" || return 1
    $PYTHON $MODEL_ROOT/preprocessing/calculate_min_max_state.py "$TRAINING_DATA" | tee "$STATE_STATS" || return 1
    echo ""
    echo "Results saved to $ACTION_STATS and $STATE_STATS"
fi
echo ""

echo ""
echo "--- Step 5: Pretokenize ---"
RYNNVLA_REPO="$HOME/RynnVLA-002/rynnvla-002"
if [ ! -d "$RYNNVLA_REPO" ]; then
    echo "ERROR: RynnVLA-002 repo not found at $RYNNVLA_REPO"
    return 1
fi
CONFIG=$($PYTHON -c "
import yaml
c = yaml.safe_load(open('$MODEL_ROOT/config.yaml'))
print(c['work_dir'] + '|' + c['task_label'] + '|' + str(c['his']) + '|' + str(c['resolution']))
")
WORK_DIR=$(echo $CONFIG | cut -d'|' -f1)
LABEL=$(echo $CONFIG | cut -d'|' -f2)
HIS=$(echo $CONFIG | cut -d'|' -f3)
RES=$(echo $CONFIG | cut -d'|' -f4)
INPUT_FILE="$(realpath "$WORK_DIR/conversations/libero_${LABEL}_his_${HIS}_train_img_state_abs_ck_1_${RES}.json")"
OUTPUT_DIR="$(realpath -m "$WORK_DIR/tokens/vla_data")"
TOKENIZER="$HOME/RynnVLA-002/rynnvla-002/ckpts/chameleon/base_model"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Tokens already exist at $OUTPUT_DIR — skipping. Delete to rerun."
else
    cd "$RYNNVLA_REPO/data_lerobot"
    $PYTHON pretoken_lerobot_state.py \
        --input_file "$INPUT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --resolution "$RES" \
        --tokenizer_path "$TOKENIZER" || { cd - > /dev/null; return 1; }
    cd - > /dev/null
fi
echo ""

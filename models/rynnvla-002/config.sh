export VENV=models/rynnvla-002/rynnvla002-venv
export INPUT_DIR=my_data/inpt_data
export WORK_DIR=my_data/model_data

# Must be consistent across preprocessing, training, and inference
export CHUNK_SIZE=20       # number of future action steps per chunk (--time_horizon at training)
export ACTION_DIM=6        # SO-101 has 6 joints
export RESOLUTION=256      # image resolution
export HIS=1               # number of history frames

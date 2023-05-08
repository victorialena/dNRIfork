# GPU=0 # Set to whatever GPU you want to use
# CUDA_VISIBLE_DEVICES=$GPU python -u dnri/experiments/ind_experiment.py [etc]

DATA_PATH='data/ind_processed/'
WORKING_DIR="logs/ind/dnri/"

SEED=42
NUM_EDGE_TYPES=4



ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64 --encoder_normalize_mode normalize_all --normalize_inputs"
DECODER_ARGS="--decoder_hidden 256"
HIDDEN_ARGS="--rnn_hidden 64"
PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types $NUM_EDGE_TYPES $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
TRAINING_ARGS="--batch_size 8 --sub_batch_size 1 --val_batch_size 1 --lr 5e-4 --use_adam --num_epochs 400 --lr_decay_factor 0.5 --lr_decay_steps 200 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --train_data_len 100"


# WORKING_DIR="logs/ind/dnri/"
# python -u dnri/experiments/ind_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS

WORKING_DIR="logs/ind/dvae/"
python -u dnri/experiments/ind_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS --edge_features

# LOAD_FOLDER="05-07-23_exp1"
# python -u dnri/experiments/ind_experiment.py --gpu --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS --load_folder $LOAD_FOLDER
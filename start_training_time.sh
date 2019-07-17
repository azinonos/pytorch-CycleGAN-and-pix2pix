#!/bin/bash

# PARAMETERS
# DATA='brain_slices_0-90d_coregistered_atlas_tumourOnly'
EXP_NAME='time_prediction_01'
MODEL='time_predictor'
INPUT_NC=1
OUTPUT_NC=1
DATASET_MODE='brain'

# TRAINING
python train_time.py --dataroot /Users/azinonos/Desktop/slices_atlasFixed_tumourOnly_allDates/ --dataset_mode $DATASET_MODE --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --gpu_ids -1

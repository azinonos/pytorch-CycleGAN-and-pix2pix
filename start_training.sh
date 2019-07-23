#!/bin/bash

# PARAMETERS
DATA='brain_slices_0-90d_coregistered_atlas_tumourOnly'
EXP_NAME='brain_coreg_atlas_tumourOnly_0-90d_pix2pix_03'
MODEL='pix2pix'
INPUT_NC=1
OUTPUT_NC=1

# TRAINING
python train.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC
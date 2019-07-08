#!/bin/bash

# PARAMETERS
DATA='brain_slices_0-90d_coregistered_atlas_tumourOnly'
EXP_NAME='brain_coreg_atlas_tumourOnly_0-90d_pix2pix_03'
MODEL='pix2pix'
NUM_TEST=1000
INPUT_NC=1
OUTPUT_NC=1

# TESTING
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase train
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase test

# PREPARE FILES FOR DOWNLOAD
cd results/$EXP_NAME/
zip -r test_results.zip train_latest test_latest
cd ../../

#!/bin/bash

# PARAMETERS
DATA='brain_slices_0-90d_coregistered_CycleGAN'
EXP_NAME='brain_coreg_0-90d_cyclegan_01'
MODEL='cycle_gan'
NUM_TEST=200
INPUT_NC=1
OUTPUT_NC=1

# TESTING
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase train
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase test

# PREPARE FILES FOR DOWNLOAD
cd results/$EXP_NAME/
zip -r test_results.zip train_latest test_latest
cd ../../

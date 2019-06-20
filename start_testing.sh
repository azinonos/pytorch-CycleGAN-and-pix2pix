#!/bin/bash

# PARAMETERS
DATA='brain_slices_0-90d_unregistered'
EXP_NAME='brain_unreg_0-90d_pix2pix_01'
MODEL='pix2pix'
NUM_TEST=200
INPUT_NC=1
OUTPUT_NC=1

# TESTING
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase train
python test.py --dataroot ../brain_data/slices/$DATA/ --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase test

# PREPARE FILES FOR DOWNLOAD
zip -r results/$EXP_NAME/test_results.zip results/$EXP_NAME/train_latest results/$EXP_NAME/test_latest

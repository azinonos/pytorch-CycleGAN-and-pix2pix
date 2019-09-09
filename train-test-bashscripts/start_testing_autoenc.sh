#!/bin/bash

# PARAMETERS
DATA='brain_slices_allDates_coregistered_atlas_HM_tumourOnly'
EXP_NAME='autoenc_01'
MODEL='auto_encoder'
NUM_TEST=20000
INPUT_NC=1
OUTPUT_NC=1
DATASET_MODE='brain'

# TESTING
cd ..
python test.py --dataroot ../brain_data/slices/$DATA/ --dataset_mode $DATASET_MODE --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase train --ndf 16 --batch_size 128 --eval > log.txt
python test.py --dataroot ../brain_data/slices/$DATA/ --dataset_mode $DATASET_MODE --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase test --ndf 16 --batch_size 128 --eval >> log.txt
python test.py --dataroot ../brain_data/slices/$DATA/ --dataset_mode $DATASET_MODE --name $EXP_NAME  --model $MODEL --direction AtoB --input_nc $INPUT_NC --output_nc $OUTPUT_NC --num_test $NUM_TEST --phase masks --ndf 16 --batch_size 128 --eval >> log.txt

# PREPARE FILES FOR DOWNLOAD
cd results/$EXP_NAME/
rm -r masks_latest/images/*_fake_*
#zip -r test_results.zip train_latest test_latest masks_latest
cd ../../

#!/bin/bash

# ./main [TRAIN dataset path] [TEST dataset path] [DIM] [ITER] [Learning Rate] [Quantize]
./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 2000 20 1 100
./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 4000 20 1 100
./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 6000 20 1 100
./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 8000 20 1 100
./main datasets/UCIHAR/UCIHAR_train.choir_dat datasets/UCIHAR/UCIHAR_test.choir_dat 10000 20 1 100

./main datasets/cardiotocography/cardiotocography_train.choir_dat datasets/cardiotocography/cardiotocography_test.choir_dat 2000 20 1 100
./main datasets/cardiotocography/cardiotocography_train.choir_dat datasets/cardiotocography/cardiotocography_test.choir_dat 4000 20 1 100
./main datasets/cardiotocography/cardiotocography_train.choir_dat datasets/cardiotocography/cardiotocography_test.choir_dat 6000 20 1 100
./main datasets/cardiotocography/cardiotocography_train.choir_dat datasets/cardiotocography/cardiotocography_test.choir_dat 8000 20 1 100
./main datasets/cardiotocography/cardiotocography_train.choir_dat datasets/cardiotocography/cardiotocography_test.choir_dat 10000 20 1 100

./main datasets/face/face_train.choir_dat datasets/face/face_test.choir_dat 2000 20 1 100
./main datasets/face/face_train.choir_dat datasets/face/face_test.choir_dat 4000 20 1 100
./main datasets/face/face_train.choir_dat datasets/face/face_test.choir_dat 6000 20 1 100
./main datasets/face/face_train.choir_dat datasets/face/face_test.choir_dat 8000 20 1 100
./main datasets/face/face_train.choir_dat datasets/face/face_test.choir_dat 10000 20 1 100

./main datasets/isolet/isolet_train.choir_dat datasets/isolet/isolet_test.choir_dat 2000 20 1 100
./main datasets/isolet/isolet_train.choir_dat datasets/isolet/isolet_test.choir_dat 4000 20 1 100
./main datasets/isolet/isolet_train.choir_dat datasets/isolet/isolet_test.choir_dat 6000 20 1 100
./main datasets/isolet/isolet_train.choir_dat datasets/isolet/isolet_test.choir_dat 8000 20 1 100
./main datasets/isolet/isolet_train.choir_dat datasets/isolet/isolet_test.choir_dat 10000 20 1 100

echo "Begin EMG for D = 2000"
./main datasets/EMG/EMG_1_train.choir_dat datasets/EMG/EMG_1_test.choir_dat 2000 20 1 100
./main datasets/EMG/EMG_2_train.choir_dat datasets/EMG/EMG_2_test.choir_dat 2000 20 1 100
./main datasets/EMG/EMG_3_train.choir_dat datasets/EMG/EMG_3_test.choir_dat 2000 20 1 100
./main datasets/EMG/EMG_4_train.choir_dat datasets/EMG/EMG_4_test.choir_dat 2000 20 1 100
./main datasets/EMG/EMG_5_train.choir_dat datasets/EMG/EMG_5_test.choir_dat 2000 20 1 100
echo "Begin EMG for D = 4000"
./main datasets/EMG/EMG_1_train.choir_dat datasets/EMG/EMG_1_test.choir_dat 4000 20 1 100
./main datasets/EMG/EMG_2_train.choir_dat datasets/EMG/EMG_2_test.choir_dat 4000 20 1 100
./main datasets/EMG/EMG_3_train.choir_dat datasets/EMG/EMG_3_test.choir_dat 4000 20 1 100
./main datasets/EMG/EMG_4_train.choir_dat datasets/EMG/EMG_4_test.choir_dat 4000 20 1 100
./main datasets/EMG/EMG_5_train.choir_dat datasets/EMG/EMG_5_test.choir_dat 4000 20 1 100
echo "Begin EMG for D = 6000"
./main datasets/EMG/EMG_1_train.choir_dat datasets/EMG/EMG_1_test.choir_dat 6000 20 1 100
./main datasets/EMG/EMG_2_train.choir_dat datasets/EMG/EMG_2_test.choir_dat 6000 20 1 100
./main datasets/EMG/EMG_3_train.choir_dat datasets/EMG/EMG_3_test.choir_dat 6000 20 1 100
./main datasets/EMG/EMG_4_train.choir_dat datasets/EMG/EMG_4_test.choir_dat 6000 20 1 100
./main datasets/EMG/EMG_5_train.choir_dat datasets/EMG/EMG_5_test.choir_dat 6000 20 1 100
echo "Begin EMG for D = 8000"
./main datasets/EMG/EMG_1_train.choir_dat datasets/EMG/EMG_1_test.choir_dat 8000 20 1 100
./main datasets/EMG/EMG_2_train.choir_dat datasets/EMG/EMG_2_test.choir_dat 8000 20 1 100
./main datasets/EMG/EMG_3_train.choir_dat datasets/EMG/EMG_3_test.choir_dat 8000 20 1 100
./main datasets/EMG/EMG_4_train.choir_dat datasets/EMG/EMG_4_test.choir_dat 8000 20 1 100
./main datasets/EMG/EMG_5_train.choir_dat datasets/EMG/EMG_5_test.choir_dat 8000 20 1 100
echo "Begin EMG for D = 10000"
./main datasets/EMG/EMG_1_train.choir_dat datasets/EMG/EMG_1_test.choir_dat 10000 20 1 100
./main datasets/EMG/EMG_2_train.choir_dat datasets/EMG/EMG_2_test.choir_dat 10000 20 1 100
./main datasets/EMG/EMG_3_train.choir_dat datasets/EMG/EMG_3_test.choir_dat 10000 20 1 100
./main datasets/EMG/EMG_4_train.choir_dat datasets/EMG/EMG_4_test.choir_dat 10000 20 1 100
./main datasets/EMG/EMG_5_train.choir_dat datasets/EMG/EMG_5_test.choir_dat 10000 20 1 100

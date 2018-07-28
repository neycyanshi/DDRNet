#!/usr/bin/env bash

# e.g. checkpoint_dir=../log/cscd/noBN_L1_sd100_B16/

expm=$1
sampledir=`realpath sample/$expm`
echo "sampledir: "$sampledir
time=`date +"%m-%d-%H-%M"`
logfile=${sampledir}/log_$time.txt
mkdir -p $sampledir
touch $logfile

cd src

unbuffer python evaluate.py \
    --dnnet=convResnet \
    --dtnet=hypercolumn \
    --sample_dir=$sampledir \
    --checkpoint_dir=../log/cscd/noBN_L1_sd100_B16/ \
    --csv_path=../dataset/test.csv \
    --low_thres=500 \
    --up_thres=3000 \
    --image_size=400 \
    --gpu \
2>&1 | tee $logfile

#!/bin/bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 cifar10"
    echo "    bash $0 mini-imagenet"
    exit
fi

set -euo pipefail
# source setup.sh

dataset=$1


# eval fewshot
reader_types="inverse knn softmax"
nshots="1 4 16 64 256"
for reader_type in $reader_types; do

    mkdir -p exp_vision/eval
    logfile=exp_vision/eval/${dataset}.${reader_type}.log
    rm $logfile -f

    for nshot in $nshots; do
        echo "Running ${nshot}-shot ${reader_type} experiments on ${dataset} ..."

        if [ "$nshot" == "full" ]; then
            job=full
            nshot=1
        else
            job=fewshot
        fi
    
        uv run -m vision.eval --job $job --dataset $dataset --classifier memory --reader_type $reader_type --num_shot $nshot >> $logfile

        echo Log to $logfile
    done
done

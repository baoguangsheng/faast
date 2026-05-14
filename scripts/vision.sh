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

# run command
# train linear model with full data
uv run -m vision.train --num_epochs 20  --num_workers 12  --dataset $dataset

# train linear model with fewshot
uv run -m vision.train_fewshot --num_epochs 20  --num_workers 12  --dataset $dataset --num_shot 5

# eval fewshot
nshots = "1 5"
datasets = "cifar10 mini-imagenet"
reader_types = "knn softmax inverse"
for dataset in $datasets; do
    for reader_type in $reader_types; do
        for nshot in $nshots; do
            echo "Running ${nshot}-shot ${reader_type} experiments on ${dataset} ..."

            uv run -m vision.eval --job fewshot --dataset $dataset --classifier memory --reader_type $reader_type --num_shot $nshot
        done
    done
done

#!/bin/bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 finished_models/memory-gpt2-xl-ls15-next-token sst2"
    exit
fi

set -euo pipefail
source setup.sh


# run command
model_path=$1
dataset=$2
nsamples=5000


# set num_shots to "1 2 full" for imdb
# set num_shots to "1 5 full" for others
if [ "$dataset" == "imdb" ]; then
    num_shots="full 1 2"
else
    num_shots="full 1 5"
fi


logfile=exp_eval/$(basename ${model_path}).${dataset}.log
rm $logfile -f

for num_shot in $num_shots; do
    echo "Evaluating ${num_shot}-shot on ${dataset} ..."

    if [ "$num_shot" == "full" ]; then
        job=full
        num_shot=1
    else
        job=fewshot
    fi

    uv run -m llm.eval_fewshot --model_path $model_path --dataset $dataset --job $job --num_shot $num_shot --nsamples $nsamples >> $logfile
    
    echo Log to $logfile
done


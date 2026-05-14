#!/bin/bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 linear gpt2 1"
    echo "    bash $0 memory gpt2 1 64 0.9"
    exit
fi

set -euo pipefail
source setup.sh


# run command
model_type=$1
base_model=$2
num_layers=$3

if [ "$model_type" == "linear" ]; then
    model_path=exp_llm/models/${model_type}-${base_model}-ls${num_layers}
    output_path=exp_llm/output/${model_type}-${base_model}-ls${num_layers}
else
    max_memory_size=$4  # in MB
    update_discount=$5  # only for memory model
    model_path=exp_llm/models/${model_type}-${base_model}-ls${num_layers}-ms${max_memory_size}-ud${update_discount}
    output_path=exp_llm/output/${model_type}-${base_model}-ls${num_layers}-ms${max_memory_size}-ud${update_discount}
fi

# cp ${output_path}/model.safetensors* ${model_path}/. -f
# cp ${output_path}/model*.safetensors ${model_path}/. -f
# uv run -m llm.eval_ppl --model_path ${model_path} --learn_samples 2000

checkpoints="1000 2000 3000 6000 10000 20000"
for checkpoint in $checkpoints; do
    checkpoint_path=${output_path}/checkpoint-${checkpoint}
    while true; do
    if [ ! -d ${checkpoint_path} ]; then
        echo "Checkpoint ${checkpoint_path} does not exist, sleep 300s."
        sleep 300
        continue
    else
        echo "Found checkpoint ${checkpoint_path}."
        break
    fi
    done
    echo "Evaluating checkpoint ${checkpoint_path} ..."
    cp ${checkpoint_path}/model.safetensors* ${model_path}/. -f
    cp ${checkpoint_path}/model*.safetensors ${model_path}/. -f
    uv run -m llm.eval_ppl --model_path ${model_path} --learn_samples 20000
done



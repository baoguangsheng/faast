#!/bin/bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 2 linear gpt2 1"
    echo "    bash $0 4 memory gpt2 1 64 0.9"
    exit
fi

set -euo pipefail
source setup.sh

# run command
nprocs=$1
model_type=$2
base_model=$3
num_layers=$4


if [ "$model_type" == "linear" ]; then
    echo "Running linear ${base_model} with ${nprocs} processes ..."
    # run linear gpt2
    # - initialize exp_llm/models/linear-gpt2 with pretrained gpt2
    # - run linear gpt2 training

    model_path=exp_llm/models/${model_type}-${base_model}-ls${num_layers}
    output_path=exp_llm/output/${model_type}-${base_model}-ls${num_layers}

    rm ${model_path} -rf
    rm ${output_path} -rf

    uv run -m llm.memory.linear_causal_lm --base_model $base_model --num_layers $num_layers --model_path ${model_path}

    ACCELERATE_LOG_LEVEL=info uv run accelerate launch --config_file llm/configs/ddp.yaml --num_processes ${nprocs} \
        -m llm.train_linear --config llm/configs/config_linear-gpt2.yaml --model_name_or_path ${model_path} \
        --output_dir ${output_path}

    cp ${output_path}/model.safetensors ${model_path}/model.safetensors -f
else
    echo "Running memory ${base_model} with ${nprocs} processes ..." 
    # run memory gpt2
    # - initialize exp_llm/models/memory-gpt2 with pretrained gpt2
    # - run memory gpt2 training

    max_memory_size=$5  # in K tokens
    update_discount=$6  # only for memory model
    model_path=exp_llm/models/${model_type}-${base_model}-ls${num_layers}-ms${max_memory_size}-ud${update_discount}
    output_path=exp_llm/output/${model_type}-${base_model}-ls${num_layers}-ms${max_memory_size}-ud${update_discount}

    rm ${model_path} -rf
    rm ${output_path} -rf

    uv run -m llm.memory.memory_causal_lm --base_model $base_model --num_layers $num_layers --model_path ${model_path} --reader_update_discount ${update_discount}

    ACCELERATE_LOG_LEVEL=info uv run accelerate launch --config_file llm/configs/ddp.yaml --num_processes ${nprocs} \
        -m llm.train_memory --config llm/configs/config_memory-gpt2.yaml --model_name_or_path ${model_path} \
        --output_dir ${output_path} --max_memory_size ${max_memory_size}

    cp ${output_path}/model.safetensors ${model_path}/model.safetensors -f
fi




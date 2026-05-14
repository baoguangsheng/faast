#!/bin/bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 gpt2"
    echo "    bash $0 qwen2"
    echo "    bash $0 llama"
    exit
fi

set -euo pipefail
source handbook/bin/activate

# run command
model_type=$1


if [ "$model_type" == "gpt2" ]; then
    echo "Preparing GPT-2 data ..."
    # prepare gpt2 data
    uv run -m llm.data --config llm/configs/config_data_gpt2.yaml --use_cpu
elif [ "$model_type" == "qwen2" ]; then
    echo "Preparing Qwen-2 data ..."
    # prepare qwen2 data
    uv run -m llm.data --config llm/configs/config_data_qwen2.yaml --use_cpu
elif [ "$model_type" == "llama" ]; then
    echo "Preparing Llama data ..."
    # prepare llama data
    uv run -m llm.data --config llm/configs/config_data_llama.yaml --use_cpu
else
    echo "Unknown model type: ${model_type}"
    exit 1
fi

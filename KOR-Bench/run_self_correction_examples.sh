#!/bin/bash

set -x
# Prepare repository and environment
# git clone https://github.com/KOR-Bench/KOR-Bench.git
# cd ./KOR-Bench
# pip install -r requirements.txt

export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode self-correction --model_name Qwen2.5-0.5B-Instruct --output_dir results/self_correction --batch_size 1000 --use_accel
sleep 5

# Run chat model inference with hf transformers examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode self-correction --model_name Qwen2.5-0.5B-Instruct --output_dir results/self_correction --batch_size 16
sleep 5

# Run inference with openai api examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode self-correction --model_name gpt-4o --output_dir results/self_correction --num_worker 64
sleep 5

# Run evaluation
python eval/eval.py results/self_correction eval/results/self_correction eval/results_self_correction.csv


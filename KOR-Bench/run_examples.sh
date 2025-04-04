#!/bin/bash

set -x
# Prepare repository and environment
# git clone https://github.com/KOR-Bench/KOR-Bench.git
# cd ./KOR-Bench
# pip install -r requirements.txt

# Set HF_HOME to custom folder
# export HF_HOME=$(pwd)/cache

export PYTHONPATH=$(pwd)

# Run chat model inference with accelerate framework examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode zero-shot --model_name Qwen2.5-0.5B-Instruct --output_dir results --batch_size 250 --use_accel
sleep 5

# Run chat model inference with hf transformers examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode zero-shot --model_name Qwen2.5-0.5B-Instruct --output_dir results --batch_size 16
sleep 5

# Run base model inference with accelerate framework examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode three-shot --model_name Qwen2.5-0.5B --output_dir results --batch_size 250 --use_accel
sleep 5

# Run base model inference with hf transformers examples
python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode three-shot --model_name Qwen2.5-0.5B --output_dir results --batch_size 16
sleep 5

# Run inference with openai api examples
# python infer/infer.py --config config/config.yaml --split logic cipher counterfactual operation puzzle --mode zero-shot --model_name gpt-4o --output_dir results --num_worker 64
# sleep 5

# Run evaluation
python eval/eval.py results eval/results eval/results.csv

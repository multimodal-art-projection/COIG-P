#!/bin/bash

set -x
# Prepare repository and environment
# git clone https://github.com/KOR-Bench/KOR-Bench.git
# cd ./KOR-Bench
# pip install -r requirements.txt

export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config.yaml --split cipher --mode subquestions --model_name Qwen2.5-0.5B-Instruct --output_dir results/cipher_subquestions --batch_size 250 --use_accel
sleep 5

# Run chat model inference with hf transformers examples
python infer/infer.py --config config/config.yaml --split cipher --mode subquestions --model_name Qwen2.5-0.5B-Instruct --output_dir results/cipher_subquestions --batch_size 16
sleep 5

# Run inference with openai api examples
python infer/infer.py --config config/config.yaml --split cipher --mode subquestions --model_name gpt-4o --output_dir results/cipher_subquestions --num_worker 64
sleep 5

# Run evaluation
python eval/eval.py results/cipher_subquestions eval/results/cipher_subquestions eval/results_cipher_subquestions.csv

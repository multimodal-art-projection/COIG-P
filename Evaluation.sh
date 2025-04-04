#!/bin/bash

MODEL_NAME=Qwen2.5-7B

run_KOR() {
   set -x  
   source /map-vepfs/miniconda3/bin/activate
   conda activate zk_infer || { echo "conda激活失败"; exit 1; }  # 错误检查  
   cd KOR-Bench || exit 1  # 目录检查 
   export PYTHONPATH=$(pwd)
   
   CUDA_VISIBLE_DEVICES=4,5,6,7 python infer/infer.py \
     --config config/config.yaml \
     --split "logic,cipher,counterfactual,operation,puzzle" \
     --mode zero-shot \
     --model_name "$MODEL_NAME" \
     --output_dir results/ \
     --batch_size 200 \
     --use_accel || { echo "推理失败"; exit 1; }  # 错误检查

   cd ..
   cp -r "./KOR-Bench/results/${MODEL_NAME}_AlignBench.jsonl" \
         "./AlignBench/data/model_answer/${MODEL_NAME}.jsonl"
}

run_AlignBench(){
    cd AlignBench || exit 1  # 目录检查
    python judge.py\
        --config-path config/multi-dimension.json\
        --model-name $MODEL_NAME\
        --parallel 2 || { echo "评测失败"; exit 1; }

    python show_result.py \
        --input-dir data/judgment \
        --ques-file data/data_v1.1_release.jsonl \
        --save-file data/results/results.xlsx || { echo "结果生成失败"; exit 1; }
}

run_KOR
run_AlignBench
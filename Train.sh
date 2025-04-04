#!/bin/bash
conda activate llama_factory

cd ./LLaMA-Factory

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

torchrun --nproc-per-node $GPUS_PER_NODE \
    --master-addr $MASTER_ADDR \
    --node-rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    src/train.py --model_name_or_path BAAI/Infinity-Instruct-3M-0625-Qwen2-7B\
        --trust_remote_code true\
        --stage dpo\
        --do_train true\
        --finetuning_type full\
        --deepspeed  examples/deepspeed/ds_z3_config.json\
        --dataset  dpo_coig_pre\
        --template  qwen\
        --cutoff_len  6000\
        --max_samples  702398\
        --overwrite_cache  true\
        --preprocessing_num_workers  16\
        --output_dir  saves/Infinity-Instruct-3M-0625-Qwen2-7B/full/COIG-P\
        --logging_steps  10\
        --save_steps  2000\
        --plot_loss  true\
        --overwrite_output_dir  true\
        --per_device_train_batch_size  1\
        --gradient_accumulation_steps  2\
        --learning_rate  1.0e-6\
        --pref_beta 0.2\
        --pref_ftx 0.1\
        --num_train_epochs  1.0\
        --lr_scheduler_type  cosine\
        --warmup_ratio  0.1\
        --bf16  true\
        --report_to wandb\
        --ddp_timeout  180000000


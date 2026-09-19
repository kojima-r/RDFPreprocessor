#!/bin/bash
# GPU 0 が混雑している場合は 1 を使う(H200 NVL 144GB)
# 占有状況は `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` で確認できる
CUDA_VISIBLE_DEVICES=1 python 30kg.py \
    --data_path ./data06_uniq/chembl.graph.tsv \
    --model transe \
    --output_dir output_kg/chembl_transe \
    --batch_size 8192 \
    --num_negatives 128 \
    --eval_batch_size 256 \
    --entity_chunk_size 40000 \
    --eval_every 5 \
    --use_amp true

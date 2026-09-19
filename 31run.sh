#!/bin/bash
# 空き GPU を `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` で確認して指定する。
# このマシンは 2GPU 構成 (0, 1)。元の `=2` は存在しないので 1 を使う。
#
# 注意: --metadata_path は {"num_entities": N, "num_relations": M} を含む JSON が必須。
# data10/split_stats.json はこれらのキーを持たないので、別途 bgee 用 / chembl 用に
# 整数カウントを書き込んだ JSON を用意してから差し替えること(無いと起動時にエラー終了する)。
CUDA_VISIBLE_DEVICES=1 python 31kg_stream.py \
    --train_paths ./data10/bgee.graph.train.tsv \
    --valid_paths ./data10/bgee.graph.valid.tsv \
    --test_paths  ./data10/bgee.graph.test.tsv \
    --output_dir data10_output \
    --metadata_path ./data10/split_stats.json \
    --batch_size 8192 \
    --num_negatives 128 \
    --eval_batch_size 256 \
    --entity_chunk_size 40000 \
    --num_workers 4 \
    --prefetch_factor 4 \
    --use_amp true \
    --sparse_entity false

#CUDA_VISIBLE_DEVICES=1 python 31kg_stream.py \
#    --train_paths ./data10/chembl.graph.train.tsv \
#    --valid_paths ./data10/chembl.graph.valid.tsv \
#    --test_paths  ./data10/chembl.graph.test.tsv \
#    --output_dir data10_output \
#    --metadata_path ./test_chembl_info.json \
#    --batch_size 8192 \
#    --num_negatives 128 \
#    --eval_batch_size 256 \
#    --entity_chunk_size 40000 \
#    --num_workers 4 \
#    --prefetch_factor 4 \
#    --use_amp true \
#    --sparse_entity false

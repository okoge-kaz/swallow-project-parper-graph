#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 12288 \
  --num-layers 96 \
  --vocab-size 100000 \
  --batch-size 1536 \
  --seq-len 4096 \
  --intermediate-size 33792 \
  --num-query-groups 16 \
  --iteration 1 \
  --num-attention-heads 96

# 1 iteration: FLOPS: 6.75069737957956e+18

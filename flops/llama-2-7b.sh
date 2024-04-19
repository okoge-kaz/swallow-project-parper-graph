#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 4096 \
  --num-layers 32 \
  --vocab-size 32000 \
  --batch-size 2048 \
  --seq-len 4096 \
  --intermediate-size 11008 \
  --num-query-groups 32 \
  --iteration 1 \
  --num-attention-heads 32

# FLOPS: 

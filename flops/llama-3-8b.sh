#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 4096 \
  --num-layers 32 \
  --vocab-size 128256 \
  --batch-size 1024 \
  --seq-len 8192 \
  --intermediate-size 14336 \
  --num-query-groups 8 \
  --iteration 1 \
  --num-attention-heads 32

# FLOPS: 4.1994307306625434e+17

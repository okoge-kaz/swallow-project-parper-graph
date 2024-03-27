#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 8192 \
  --num-layers 80 \
  --vocab-size 43176 \
  --batch-size 1024 \
  --seq-len 4096 \
  --intermediate-size 28672 \
  --num-query-groups 8 \
  --iteration 25000 \
  --num-attention-heads 64

# FLOPS: 4.920972866884731e+22

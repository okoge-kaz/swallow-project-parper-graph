#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 4096 \
  --num-layers 32 \
  --vocab-size 100000 \
  --batch-size 1024 \
  --seq-len 4096 \
  --intermediate-size 14336 \
  --num-query-groups 8 \
  --iteration 1 \
  --num-attention-heads 32 \
  --swiglu \
  --num-experts 8 \
  --moe-router-topk 2

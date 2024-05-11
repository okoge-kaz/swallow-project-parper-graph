#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 8192 \
  --num-layers 48 \
  --vocab-size 100000 \
  --batch-size 1024 \
  --seq-len 4096 \
  --intermediate-size 22016 \
  --num-query-groups 8 \
  --iteration 1 \
  --num-attention-heads 64 \
  --swiglu \
  --num-experts 8 \
  --moe-router-topk 2

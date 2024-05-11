#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 8192 \
  --num-layers 80 \
  --vocab-size 100000 \
  --batch-size 1024 \
  --seq-len 4096 \
  --intermediate-size 28672 \
  --num-query-groups 8 \
  --iteration 1 \
  --num-attention-heads 64 \
  --swiglu

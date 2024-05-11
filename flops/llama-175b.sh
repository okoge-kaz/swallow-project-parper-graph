#!/bin/bash

python flops/llama-2-architecutre.py \
  --hidden-size 12288 \
  --num-layers 96 \
  --vocab-size 100000 \
  --batch-size 1728 \
  --seq-len 4096 \
  --intermediate-size 38464 \
  --num-query-groups 16 \
  --iteration 1 \
  --num-attention-heads 96 \
  --swiglu

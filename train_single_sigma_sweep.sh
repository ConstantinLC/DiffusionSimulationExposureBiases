#!/bin/bash

SIGMAS=(-2.9 -2.8 -2.7 -2.6 -2.5 -2.4 -2.3 -2.2 -2.1 -2.0 -1.9)
N_GPUS=8

mkdir -p logs

echo "=== Launching ${#SIGMAS[@]} jobs across $N_GPUS GPUs (2 per GPU) ==="
for i in "${!SIGMAS[@]}"; do
  sigma=${SIGMAS[$i]}
  gpu=$((i % N_GPUS))
  echo "  GPU $gpu <- sigma $sigma"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    '+experiment=ks_single_sigma' \
    "model.log_sigma_min=${sigma}" \
    "checkpoint_dir=./checkpoints/KuramotoSivashinsky/single_sigma/sigma_${sigma}" \
    > "logs/single_sigma_${sigma}.log" 2>&1 &
done

wait
echo "=== All done ==="

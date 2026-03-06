#!/bin/bash
# Like train_single_sigma_sweep.sh but with a separate Unet per noise level
# (multi_unet=True: one Unet for the cold-start step, one for the sigma_min step).

SIGMAS=(-2.5 -2.4 -2.3 -2.2 -2.1 -2.0 -1.9 -1.8 -1.7 -1.6 -1.5)
N_GPUS=8

mkdir -p logs

echo "=== Launching ${#SIGMAS[@]} multi-unet refiner jobs across $N_GPUS GPUs ==="
for i in "${!SIGMAS[@]}"; do
  sigma=${SIGMAS[$i]}
  gpu=$((i % N_GPUS))
  echo "  GPU $gpu <- sigma $sigma"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    '+experiment=ks_single_sigma' \
    "model.log_sigma_min=${sigma}" \
    "+model.multi_unet=true" \
    "checkpoint_dir=./checkpoints/KuramotoSivashinsky/single_sigma_multi_unet/sigma_${sigma}" \
    > "logs/single_sigma_multi_unet_${sigma}.log" 2>&1 &
done

wait
echo "=== All done ==="

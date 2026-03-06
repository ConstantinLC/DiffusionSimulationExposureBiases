#!/bin/bash

mkdir -p logs

echo "=== Training standalone Unet ==="
python train.py \
  '+experiment=ks_unet' \
  > "logs/unet.log" 2>&1

echo "=== Done ==="

#!/usr/bin/bash

task=(stack_three_d1 threading_d2 coffee_d2 three_piece_assembly_d2)
n_runs=${#task[@]}
cd ..

for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=train_diffusion_unet task_name=${task[$i]} training.device=cuda:${i} \
  > ./logs/$(date '+%m%d%H%M')_dp_${task[$i]}_train.txt 2>&1 & sleep 10m
done

wait
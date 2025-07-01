#!/usr/bin/bash

var=2
n_demo=100
seed=42
task=(three_piece_assembly_d${var} square_d${var} coffee_d${var} threading_d${var})
n_runs=${#task[@]}
canonicalize=true
cd ..

for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=equibot_ddpm_II \
  task_name=${task[$i]} training.device=cuda:${i} \
  logging.project=EquiBot \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.canonicalize=${canonicalize} \
  logging.name=$(date '+%m%d%H%M')_equibot_c${canonicalize}_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d%H%M')_equibot_c${canonicalize}_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

wait
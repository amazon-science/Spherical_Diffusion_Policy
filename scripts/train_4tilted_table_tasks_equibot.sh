#!/usr/bin/bash

var=4
project=EquiBot

n_demo=100
task=(three_piece_assembly_d${var} square_d${var} coffee_d${var} threading_d${var})
n_runs=${#task[@]}
cd ..

g=0
for seed in 7 100;
do

# ours equibot
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]} on GPU${g}."
  nohup python train.py --config-name=equibot_ddpm_II \
  logging.project=${project} \
  task_name=${task[$i]} training.device=${g} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  logging.name=$(date '+%m%d_%H%M')_equibot_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_equibot_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
  ((g+=1))
done

done

wait
#!/usr/bin/bash

task=(kitchen_d1  three_piece_assembly_d2  pick_place_d0 coffee_preparation_d1)
n_runs=${#task[@]}
cd ..

n_demo=100
seed=7
# ours
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  task_name=${task[$i]} training.device=cuda:${i} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  logging.project=SDP_V2_MimicGen \
  logging.name=$(date '+%m%d_%H%M')_SDP_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_SDP_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

wait
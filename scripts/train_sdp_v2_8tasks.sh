#!/usr/bin/bash

task=(stack_d1 threading_d2 coffee_d2 hammer_cleanup_d1 nut_assembly_d0 stack_three_d1 square_d2 mug_cleanup_d1)
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
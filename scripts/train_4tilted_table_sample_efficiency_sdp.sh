#!/usr/bin/bash

var=3
project=SDP_Sample_Efficiency

seed=42
task=(three_piece_assembly_d${var} square_d${var} coffee_d${var} threading_d${var})
n_runs=${#task[@]}
cd ..

g=0
for n_demo in 316;
do

# sdp v2
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]} on GPU${g}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  logging.project=${project} \
  task_name=${task[$i]} training.device=${g} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  logging.name=$(date '+%m%d_%H%M')_SDP_V2_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_SDP_V2_${task[$i]}_${seed}_train.txt 2>&1 & sleep 10m
  ((g+=1))
done

done

wait
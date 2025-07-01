#!/usr/bin/bash

var=2
task=(three_piece_assembly_d${var} square_d${var})
#task=(coffee_d${var} threading_d${var})
n_runs=${#task[@]}
cd ..

project=SDP_ablation

# sdp
n_demo=100
seed=42

for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:${i} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.is_SFiLM=false \
  logging.name=$(date '+%m%d_%H%M')_sdp_FiLM_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_FiLM_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done


for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i}+2)) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.denoise_nn=nn \
  logging.name=$(date '+%m%d_%H%M')_sdp_d_nn_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_d_nn_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done


#for ((i=0; i<n_runs; i++));
#do
#  echo "Starting training ${task[$i]}."
#  nohup python train.py --config-name=sdp_ddpm \
#  logging.project=${project} \
#  task_name=${task[$i]} training.device=cuda:$((${i}+4)) \
#  n_demo=${n_demo} \
#  training.seed=${seed} \
#  policy.canonicalize=false \
#  policy.rad_aug=0.06 \
#  logging.name=$(date '+%m%d_%H%M')_sdp_no_relact_${task[$i]}_${n_demo}_${seed} \
#  > ./logs/$(date '+%m%d_%H%M')_sdp_no_relact_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
#done
#
#
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i}+6)) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.lmax=3 \
  policy.mmax=2 \
  logging.name=$(date '+%m%d_%H%M')_sdp_l3_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_l3_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

wait
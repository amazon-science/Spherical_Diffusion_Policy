#!/usr/bin/bash

var=3  # +-15 degrees tilt
task=(three_piece_assembly_d${var} square_d${var} coffee_d${var} threading_d${var})
n_runs=${#task[@]}
cd ..

project=SDP_V2_ablation
n_demo=100
seed=42

##------------------------------------------------------------------------------------------

## regular FiLM
#for ((i=0; i<n_runs; i++));
#do
#  echo "Starting training ${task[$i]}."
#  nohup python train.py --config-name=sdp_ddpm_5layer \
#  logging.project=${project} \
#  task_name=${task[$i]} training.device=cuda:${i} \
#  n_demo=${n_demo} \
#  training.seed=${seed} \
#  policy.FiLM_type=FiLM \
#  logging.name=$(date '+%m%d_%H%M')_sdp_FiLM_${task[$i]}_${n_demo}_${seed} \
#  > ./logs/$(date '+%m%d_%H%M')_sdp_FiLM_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
#done
#
## no SDTU
#for ((i=0; i<n_runs; i++));
#do
#  echo "Starting training ${task[$i]}."
#  nohup python train.py --config-name=sdp_ddpm_5layer \
#  logging.project=${project} \
#  task_name=${task[$i]} training.device=cuda:$((${i}+4)) \
#  n_demo=${n_demo} \
#  training.seed=${seed} \
#  policy.denoise_nn=nn \
#  logging.name=$(date '+%m%d_%H%M')_sdp_d_nn_${task[$i]}_${n_demo}_${seed} \
#  > ./logs/$(date '+%m%d_%H%M')_sdp_d_nn_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
#done

##------------------------------------------------------------------------------------------

# no relative action
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i})) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.canonicalize=false \
  policy.rad_aug=0.06 \
  logging.name=$(date '+%m%d_%H%M')_sdp_no_relact_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_no_relact_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

# degree l=1
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i}+4)) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.lmax=1 \
  policy.mmax=1 \
  logging.name=$(date '+%m%d_%H%M')_sdp_l1_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_l1_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

##------------------------------------------------------------------------------------------

# degree l=3
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i})) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.lmax=3 \
  policy.mmax=2 \
  logging.name=$(date '+%m%d_%H%M')_sdp_l3_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_l3_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

##------------------------------------------------------------------------------------------

# reg nn
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=sdp_ddpm_5layer \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:${i} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  policy.denoise_nn=reg \
  logging.name=$(date '+%m%d_%H%M')_sdp_reg_nn_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_sdp_reg_nn_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

#------------------------------------------------------------------------------------------

# cano dp3
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=dp3_cano \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:${i} \
  n_demo=${n_demo} \
  training.seed=${seed} \
  logging.name=$(date '+%m%d_%H%M')_dp3_cano_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_dp3_cano_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

# dp3 (no equ)
for ((i=0; i<n_runs; i++));
do
  echo "Starting training ${task[$i]}."
  nohup python train.py --config-name=dp3 \
  logging.project=${project} \
  task_name=${task[$i]} training.device=cuda:$((${i}+4)) \
  n_demo=${n_demo} \
  training.seed=${seed} \
  logging.name=$(date '+%m%d_%H%M')_dp3_${task[$i]}_${n_demo}_${seed} \
  > ./logs/$(date '+%m%d_%H%M')_dp3_${task[$i]}_${seed}_train.txt 2>&1 & sleep 1m
done

wait
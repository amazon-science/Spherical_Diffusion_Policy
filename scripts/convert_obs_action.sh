#!/usr/bin/bash

tasks=(square three_piece_assembly coffee threading)
variation=3
n_runs=${#tasks[@]}

cd ..
# in equidiffpo

for ((i=0; i<n_runs; i++));
do
  task=${tasks[$i]}
  echo "Starting img voxel pcd generation for ${task}."
  mkdir data/robomimic/datasets/${task}_${variation} && \
  cp /tmp/core_datasets/${task}/demo_src_${task}_task_D${variation}/demo.hdf5 data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}.hdf5 && \
  python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}.hdf5 \
  --output data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}_pc.hdf5 --num_workers=12 && \
  python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}.hdf5 \
  -o data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}_abs.hdf5 -n 12  && \
  python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}_pc.hdf5 \
  -o data/robomimic/datasets/${task}_d${variation}/${task}_d${variation}_pc_abs.hdf5 -n 12 \
   2>&1 & sleep 1s
done

wait
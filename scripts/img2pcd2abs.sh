#!/usr/bin/bash

num_workers=32

for dataset in stack_d1 stack_three_d1 square_d2 threading_d2 coffee_d2 three_piece_assembly_d2 hammer_cleanup_d1 mug_cleanup_d1 kitchen_d1 nut_assembly_d0 pick_place_d0 coffee_preparation_d1;
do python ../equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/${dataset}/${dataset}.hdf5 --output data/robomimic/datasets/${dataset}/${dataset}_pc.hdf5 --num_workers=${num_workers} > ./log/${dataset}_pc_out.txt;
done

for dataset in stack_d1 stack_three_d1 square_d2 threading_d2 coffee_d2 three_piece_assembly_d2 hammer_cleanup_d1 mug_cleanup_d1 kitchen_d1 nut_assembly_d0 pick_place_d0 coffee_preparation_d1;
do python ../equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}_pc.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_pc_abs.hdf5 -n ${num_workers} > ./log/${dataset}_abs_out.txt;
done

for dataset in stack_d1 stack_three_d1 square_d2 threading_d2 coffee_d2 three_piece_assembly_d2 hammer_cleanup_d1 mug_cleanup_d1 kitchen_d1 nut_assembly_d0 pick_place_d0 coffee_preparation_d1;
do python ../equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_abs.hdf5 -n ${num_workers} > ./log/${dataset}_abs_out.txt;
done

import wandb
import numpy as np
import scipy

# project_name = 'diffusion_policy_stack_d1'
# project_name = 'diffusion_policy_stack_three_d1'
# project_name = 'diffusion_policy_threading_d2'
# project_name = 'diffusion_policy_square_d2'
project_name = 'diffusion_policy_coffee_d2'
# project_name = 'diffusion_policy_three_piece_assembly_d2'
# project_name = 'diffusion_policy_hammer_cleanup_d1'
# project_name = 'diffusion_policy_mug_cleanup_d1'
# project_name = 'diffusion_policy_kitchen_d1'
# project_name = 'diffusion_policy_nut_assembly_d0'
# project_name = 'diffusion_policy_pick_place_d0'
# project_name = 'diffusion_policy_coffee_preparation_d1'


# run_names = [
#     'diff_c_demo100', 'equi_diff_ecnnenc_demo100_b128_rotaug',
#     'diff_c_demo200', 'equi_diff_ecnnenc_demo200_b128_rotaug',
#     'diff_c_demo500', 'equi_diff_ecnnenc_demo500_b128_rotaug',
#     'diff_c_demo1000', 'equi_diff_ecnnenc_demo1000_b128_rotaug',
#     ]
# run_names = [
#     'diff_c_demo100_rotaug', 
#     'diff_c_demo200_rotaug', 
#     'diff_c_demo500_rotaug', 
#     'diff_c_demo1000_rotaug', 
#     ]

# run_names = [
#     'equi_diff_ecnnenc_demo100_b128',
#     'equi_diff_ecnnenc_demo200_b128',
#     'equi_diff_ecnnenc_demo1000_b128',
#     'diff_c_demo100',
#     'diff_c_demo200',
#     'diff_c_demo1000',
#     ]

run_names = [
    'act_demo100',
    'act_demo200',
    'act_demo1000',
    ]

# run_names = [
#     'diff_t_demo100',
#     'diff_t_demo200',
#     'diff_t_demo1000',
#     ]

# run_names = [
#     'equi_diff_ecnnenc_demo100_b128',
#     'equi_diff_ecnnenc_demo200_b128',
#     'equi_diff_ecnnenc_demo500_b128',
#     'equi_diff_ecnnenc_demo1000_b128',
#     ]

# run_names = ['equi_diff_ecnnenc_demo200_b128_se2']

for run_name in run_names:
    api = wandb.Api()
    runs = api.runs(f"dian-bdai/{project_name}")
    # filtered_runs = [run for run in runs if run.name == run_name and run.state == 'finished']
    filtered_runs = [run for run in runs if run.name == run_name]

    mean_scores = []
    for run in filtered_runs:
        history_data = run.history(samples=200, keys=['test/mean_score'], x_axis='epoch', pandas=False)
        mean_score = np.array([entry['test/mean_score'] for entry in history_data])[:50]
        if len(mean_score) < 50:
            print(f'{run} not 50 logs, {len(mean_score)}')
            mean_score = np.pad(mean_score, [50 - len(mean_score), 0])
        mean_scores.append(mean_score)
    if len(mean_scores) < 1:
        print(f'{run_name} not 1 run')
        continue
    mean_scores = np.stack(mean_scores)

    K = 3
    M = 5
    top_k = np.partition(mean_scores, -K)[:, -K:]
    maximum = np.max(mean_scores, axis=1)
    last_m = mean_scores[:, -M:]
    # print(f'{project_name}: {run_name}: max: {np.round(maximum.mean(), decimals=3)}, last{M}: {np.round(last_m.mean(), decimals=3)}, top{K}: {np.round(top_k.mean(), decimals=3)}')
    print(f'{run_name} {mean_scores.shape[0]} runs: {np.round(maximum.mean()*100, decimals=1)}$\pm${np.round(scipy.stats.sem(maximum)*100, decimals=1)}')

print(1)
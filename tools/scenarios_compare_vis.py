import os
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from joblib import Parallel, delayed

EVAL_METRICS = [
    # 'cnt_ade_car', 'cnt_ade_pedestrian', 
    # 'cnt_fde_car', 'cnt_fde_pedestrian',  
    'fp_car', 'fp_pedestrian', 'ADE_car', 'ADE_pedestrian', 
    'FDE_car', 'FDE_pedestrian', 
    # 'MR_car', 'MR_pedestrian',
    # 'hit_car', 'hit_pedestrian',  
    'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s', 'plan_obj_col_1s', 
    'plan_obj_col_2s', 'plan_obj_col_3s', 'plan_obj_box_col_1s', 
    'plan_obj_box_col_2s', 'plan_obj_box_col_3s', 
    ]

# metrics that are better when smaller
REVERSE_EVAL_METRICS = set(
    [
    'hit_car', 'hit_pedestrian',
    'fp_car', 'fp_pedestrian', 
    'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s', 'plan_obj_col_1s', 
    'plan_obj_col_2s', 'plan_obj_col_3s', 'plan_obj_box_col_1s', 
    'plan_obj_box_col_2s', 'plan_obj_box_col_3s', 
    'MR_car', 'MR_pedestrian', 
    'ADE_car', 'ADE_pedestrian', 
    'FDE_car', 'FDE_pedestrian', 
    ]
)

def extract_subpath(full_path, keyword, default_value=None):
    path_components = full_path.split(os.sep)
    for component in path_components[::-1]:
        if keyword in component:
            return component
    return default_value

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize evaluation indicators for different scenarios')
    parser.add_argument('--dir_baseline', help='directory path to store baseline jsons', default="test/scenario_test/VAD_baseline")
    parser.add_argument('--dir_exp', help='directory path to store experiment jsons')
    parser.add_argument('--save_image_dir', help='path to save the plot image', default="test/scenario_compare")
    parser.add_argument('--eval_metrics', type=str, help='Evaluation metrics to be compared')
    parser.add_argument('--keyword', type=str, help='Keyword to extract subpath, default "VAD"', default="VAD")
    args = parser.parse_args()
    return args

def plot_metric_comparison(dir_baseline, dir_exp, save_image_path, eval_metrics, keyword):
    os.makedirs(save_image_path, exist_ok=True)
    scenario_names = []
    metrics_values_1 = []
    metrics_values_2 = []

    exp1_name = extract_subpath(dir_baseline, keyword, default_value="Experiment 1")
    exp2_name = extract_subpath(dir_exp, keyword, default_value="Experiment 2")
    
    for folder_name in os.listdir(dir_baseline):
        folder_path_1 = os.path.join(dir_baseline, folder_name)
        folder_path_2 = os.path.join(dir_exp, folder_name)
        
        if not os.path.isdir(folder_path_1) or not os.path.isdir(folder_path_2):
            continue
            
        json_path_1 = os.path.join(folder_path_1, 'evaluation_results.json')
        json_path_2 = os.path.join(folder_path_2, 'evaluation_results.json')
        
        if os.path.exists(json_path_1) and os.path.exists(json_path_2):
            with open(json_path_1, 'r') as f1, open(json_path_2, 'r') as f2:
                data_1 = json.load(f1)
                data_2 = json.load(f2)
                data_1['metric_dict'] = defaultdict(float, data_1['metric_dict'])
                data_2['metric_dict'] = defaultdict(float, data_2['metric_dict'])
                
                metrics_1 = data_1['metric_dict'][eval_metrics]
                metrics_2 = data_2['metric_dict'][eval_metrics]
                
                scenario_names.append(folder_name)
                metrics_values_1.append(metrics_1)
                metrics_values_2.append(metrics_2)
        else:
            print(f"Warning: json file not found for {folder_name}")
    
    uplift_values = [(v2 - v1) / v1 * 100 if v1 != 0 else 0 for v1, v2 in zip(metrics_values_1, metrics_values_2)]
    
    plt.figure(figsize=(15, 6))

    bar_width = 15 / (len(scenario_names) + 3) / 2
    index = range(len(scenario_names))

    ax1 = plt.gca()
    ax2 = plt.gca().twinx()
    
    ax1.bar(index, metrics_values_1, bar_width, label=exp1_name, color='#d4e1ee')
    ax1.bar([i + bar_width for i in index], metrics_values_2, bar_width, label=exp2_name, color='#2a6398')
    handles1, labels1 = ax1.get_legend_handles_labels()

    if eval_metrics in REVERSE_EVAL_METRICS:
        uplift_values = [-v for v in uplift_values]
        ax2.plot([i + bar_width / 2 for i in index], uplift_values, color='red', marker='o', label='Uplift (%) - Reversed Raw')
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylabel('Uplift (%) - Reversed Raw', color='red')
    else:
        ax2.plot([i + bar_width / 2 for i in index], uplift_values, color='green', marker='o', label='Uplift (%) - Raw')
        ax2.spines['right'].set_color('green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylabel('Uplift (%) - Raw', color='green')
    
    # add uplift number on top of the bar
    for i, v in enumerate(uplift_values):
        ax2.text(i + bar_width / 2, v, f'{v:.2f}%', ha='center', va='bottom')

    # ax2.axhline(0, color='black', linestyle='--', label='0% Uplift Line')
    
    plt.xlabel('Scenario')
    ax1.set_ylabel(eval_metrics)
    plt.title(f'Comparison of {eval_metrics} in Different Scenarios')
    plt.xticks([i + bar_width / 2 for i in index], scenario_names)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    # plt.legend()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(save_image_path, f'{eval_metrics}_comparison.png')
    plt.savefig(save_path)
    print(f'{eval_metrics} comparison image saved to {save_path}')

    return scenario_names, metrics_values_1, metrics_values_2, uplift_values

def main():
    args = parse_args()
    if not args.eval_metrics:
        eval_metrics = EVAL_METRICS
    else:
        eval_metrics = [args.eval_metrics]

    Parallel(n_jobs=len(eval_metrics))(delayed(plot_metric_comparison)(
        args.dir_baseline, args.dir_exp, args.save_image_dir, eval_metric, args.keyword) 
        for eval_metric in eval_metrics)

if __name__ == '__main__':
    main()
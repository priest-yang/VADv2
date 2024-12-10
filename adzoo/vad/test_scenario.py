import subprocess
import argparse
import os
import time
from datetime import datetime
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Redirect ann_file_test(map)')
    parser.add_argument('--script_name',help='path to test.py')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--path_to_test_pkl', type=str)
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--master_port', type=int)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


from joblib import Parallel, delayed

def process_single_file(file_info, args, result_json_path):
    filename, subdir_path, device_id, master_port = file_info
    scenario_basename = os.path.basename(subdir_path)
    result_json_path = os.path.join(result_json_path, scenario_basename)
    test_pkl_path = os.path.join(subdir_path, filename)
    test_pkl_dir = os.path.dirname(test_pkl_path)
    file_name = os.path.basename(test_pkl_path).split('.')[0]
    test_map_path = os.path.join(test_pkl_dir, f'{file_name}_map.json')
    print(f"test {file_name}")
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    print(f"CUDA_VISIBLE_DEVICES: {my_env['CUDA_VISIBLE_DEVICES']}")
    process = subprocess.Popen(
        ["torchrun", "--nproc_per_node", "1", "--master_port", str(master_port), 
         args.script_name, args.config, args.checkpoint,
         "--modify_ann_file_test", test_pkl_path, "--modify_ann_file_map", test_map_path, 
         "--json_dir", result_json_path, "--launcher", args.launcher, 
         "--eval", args.eval[0]],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, 
        env=my_env  # Pass the modified environment
    )

    # ... rest of the process handling code ...
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    error_output = process.stderr.read()
    if error_output:
        print(f"Error: {error_output.strip()}")

    process.wait()
    if process.returncode != 0:
        print(f"Process failed with return code {process.returncode}")
    
    time.sleep(5)

def combine_metrics(result_json_path):
    # Dictionaries to store sums and total weights
    total_metrics = defaultdict(float)
    total_weight = defaultdict(int)
    dataset_size = 0
    
    # Process each JSON file
    for subdir_name in os.listdir(result_json_path):
        subdir_path = os.path.join(result_json_path, subdir_name)
        json_path = os.path.join(subdir_path, 'evaluation_results.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                weight = data['dataset_size']
                dataset_size += weight
                # Add weighted metrics from each dictionary
                for dict_name in ['result_dict', 'metric_dict', 'ret_dict']:
                    if dict_name in data:
                        for metric, value in data[dict_name].items():
                            if value is not None:
                                total_metrics[f"{dict_name}/{metric}"] += value * weight
                                total_weight[f"{dict_name}/{metric}"] += weight
                
    
    # Calculate weighted means
    final_metrics = {}
    for metric, weighted_sum in total_metrics.items():
        final_metrics[metric] = weighted_sum / total_weight[metric] if metric in total_weight and total_weight[metric] > 0 else None
    
    final_metrics['dataset_size'] = dataset_size


    # revert back to original metric names
    formatted_metrics = defaultdict(dict)
    for dict_name in ['result_dict', 'metric_dict', 'ret_dict']:
        for metric, value in final_metrics.items():
            if metric.startswith(dict_name):
                formatted_metrics[dict_name][metric.split(f'{dict_name}/')[-1]] = value
    formatted_metrics['dataset_size'] = final_metrics['dataset_size']

    eval_json_path = os.path.join(result_json_path, 'combined_metrics.json')
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        # 处理numpy数组和tensor
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            return obj
        
        json.dump(formatted_metrics, f, indent=4, default=convert_to_serializable)
    print(f'Evaluation results saved to {eval_json_path}')
    
    return final_metrics

def main():
    args = parse_args()
    
    # Collect all files to process
    files_to_process = []
    device_id = -1
    master_port = args.master_port
    config_name = args.config.split('/')[-1].split('.')[0]

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_json_path = os.path.join("test", "scenario_test", config_name, time_now)
    os.makedirs(result_json_path, exist_ok=True)
    # copy config file to result_json_path
    shutil.copy(args.config, os.path.join(result_json_path, f"{config_name}.py"))

    for subdir_name in os.listdir(args.path_to_test_pkl):
        subdir_path = os.path.join(args.path_to_test_pkl, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.pkl'):
                    device_id = (device_id + 1) % args.nproc_per_node
                    master_port = master_port + 1
                    files_to_process.append((filename, subdir_path, device_id, master_port))

    # Process files in parallel
    Parallel(n_jobs=args.nproc_per_node)(
        delayed(process_single_file)(file_info, args, result_json_path)
        for file_info in files_to_process
    )

    # merge all json files and save to result_json_path
    combine_metrics(result_json_path)
    
if __name__ == '__main__':
    main()
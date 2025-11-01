import os
import sys
import time
import subprocess
import argparse
import signal
import json
from collections import deque, defaultdict
from typing import List, Dict, Any
from nni.experiment import Experiment
from tqdm import tqdm
from itemrec.port_handle import find_available_port


# --- Global state for signal handling ---
running_processes: Dict[int, Dict[str, Any]] = {}

# --- START: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
def load_memory_config(memory_file="memory_test.json"):
    """Load memory configuration from JSON file."""
    try:
        with open(memory_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Memory configuration file {memory_file} not found. Using default scheduling.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {memory_file}. Using default scheduling.")
        return {}

def calculate_optimal_trials_per_gpu(memory_config, gpu_memory_mb, safety_margin=0.1):
    """Calculate optimal number of trials per GPU for each experiment configuration."""
    optimal_configs = {}
    safe_memory = gpu_memory_mb * (1 - safety_margin)  # Leave some safety margin
    
    for config in memory_config:
        key = (config['dataset'], config['model'], config['optim'])
        memory_per_trial = config['experiment_memory_mb']
        max_trials = config['maxTrial']
        
        if memory_per_trial > 0:
            # Calculate how many trials can fit in GPU memory
            max_possible = int(safe_memory // memory_per_trial)
            # Use the minimum of what's possible and what's available
            optimal_trials = min(max_trials, max_possible)
            # Ensure at least 1 trial can run
            optimal_trials = max(1, optimal_trials)
        else:
            optimal_trials = 1  # Fallback if memory is 0 or invalid
        
        optimal_configs[key] = {
            'memory_per_trial': memory_per_trial,
            'max_trials': max_trials,
            'optimal_trials_per_gpu': optimal_trials
        }
    
    return optimal_configs

def get_sorted_experiment_configs(memory_optimal_configs) -> List[Dict[str, Any]]:
    """Generates all experiment configurations sorted by memory usage (ascending)."""
    MODELS = ("MF", "LightGCN", "XSimGCL")
    LOSSES = ("BPR", "GuidedRec", "LLPAUC", "Softmax", "AdvInfoNCE", "BSL", "PSL", "SLatK")
    DATASETS = ("amazon2014-health", "amazon2014-electronic", "amazon2014-book", "gowalla")
    
    configs = []
    for dataset in DATASETS:
        for optim in LOSSES:
            for model in MODELS:
                key = (dataset, model, optim)
                memory_info = memory_optimal_configs.get(key, {})
                
                config = {
                    "model": model,
                    "dataset": dataset,
                    "optim": optim,
                    "norm": True,
                    "fold": 1,
                    "num_epochs": 200,
                    "memory_per_trial": memory_info.get('memory_per_trial', 1000),  # Default fallback
                    "optimal_trials_per_gpu": memory_info.get('optimal_trials_per_gpu', 8),  # Default fallback
                }
                configs.append(config)
    
    # Sort by memory usage (ascending) - run memory-efficient experiments first
    configs.sort(key=lambda x: x['memory_per_trial'])
    return configs

def estimate_experiment_memory_usage(config):
    """Estimate total memory usage for an experiment."""
    # Estimate based on memory per trial and number of concurrent trials
    memory_per_trial = config.get('memory_per_trial', 1000)
    optimal_trials = config.get('optimal_trials_per_gpu', 8)
    
    # Add some overhead for the experiment itself
    experiment_overhead = 100  # MB
    return memory_per_trial * optimal_trials + experiment_overhead

def pack_experiments_to_gpus(configs, gpus, max_memory_per_gpu):
    """Pack experiments into GPUs using a bin packing algorithm."""
    # Sort configs by memory usage (descending for best-fit decreasing)
    configs_sorted = sorted(configs, key=lambda x: estimate_experiment_memory_usage(x), reverse=True)
    
    # Initialize GPU memory usage
    gpu_memory_usage = {gpu: 0 for gpu in gpus}
    gpu_assignments = defaultdict(list)
    
    for config in configs_sorted:
        config_memory = estimate_experiment_memory_usage(config)
        
        # Find the GPU with the smallest remaining capacity that can fit this config
        best_gpu = None
        min_remaining = float('inf')
        
        for gpu in gpus:
            remaining_memory = max_memory_per_gpu - gpu_memory_usage[gpu]
            if remaining_memory >= config_memory and remaining_memory < min_remaining:
                best_gpu = gpu
                min_remaining = remaining_memory
        
        # If no GPU has enough space, use the one with the most free space
        if best_gpu is None:
            best_gpu = max(gpus, key=lambda gpu: max_memory_per_gpu - gpu_memory_usage[gpu])
            print(f"Warning: Experiment requires {config_memory}MB but no GPU has enough space. "
                  f"Assigning to GPU {best_gpu} with {max_memory_per_gpu - gpu_memory_usage[best_gpu]}MB free.")
        
        # Assign config to GPU
        gpu_assignments[best_gpu].append(config)
        gpu_memory_usage[best_gpu] += config_memory
    
    return gpu_assignments, gpu_memory_usage
# --- END: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---

def cleanup(signum, frame):
    """Signal handler for graceful shutdown using the NNI API."""
    print("\nTermination signal received. Shutting down all running experiments...")
    
    for port, info in list(running_processes.items()):
        print(f"  - Stopping experiment on port {port}...")
        try:
            experiment = Experiment.connect(port)
            experiment.stop()
            print(f"    Successfully stopped NNI experiment on port {port} via API.")
        except Exception as e:
            print(f"    Could not stop NNI experiment on port {port} via API: {e}")
            print(f"    Terminating process PID {info['process'].pid} directly.")
            info['process'].terminate()

    print("Cleanup complete. Exiting.")
    sys.exit(0)

def main(args):
    signal.signal(signal.SIGINT, cleanup)

    # --- START: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
    # Load memory configuration and calculate optimal trials
    memory_config = load_memory_config()
    memory_optimal_configs = calculate_optimal_trials_per_gpu(memory_config, args.max_memory_per_gpu)
    
    # Generate sorted configurations
    all_configs = get_sorted_experiment_configs(memory_optimal_configs)
    
    # Pack experiments into GPUs
    gpu_assignments, gpu_memory_usage = pack_experiments_to_gpus(all_configs, args.gpus, args.max_memory_per_gpu)
    
    # Create a queue for each GPU
    gpu_queues = {gpu: deque(configs) for gpu, configs in gpu_assignments.items()}
    
    # Track running experiments per GPU
    gpu_running_count = {gpu: 0 for gpu in args.gpus}
    gpu_memory_used = {gpu: 0 for gpu in args.gpus}
    
    # Calculate total experiments
    total_experiments = sum(len(configs) for configs in gpu_assignments.values())
    
    print(f"--- Memory-Aware Packing Scheduler Initialized ---")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Available GPUs: {sorted(list(args.gpus))}")
    print(f"Maximum GPU memory per GPU: {args.max_memory_per_gpu} MB")
    print("GPU assignments:")
    for gpu in args.gpus:
        memory_used = gpu_memory_usage.get(gpu, 0)
        utilization = (memory_used / args.max_memory_per_gpu) * 100
        print(f"  GPU {gpu}: {len(gpu_assignments[gpu])} experiments, {memory_used:.0f}MB/{args.max_memory_per_gpu}MB ({utilization:.1f}% utilization)")
    print("----------------------------------------")
    # --- END: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
    
    next_port = args.start_port
    pbar = tqdm(total=total_experiments, desc="Overall Progress", unit="exp")
    
    while any(gpu_queues.values()) or running_processes:
        
        # --- 1. Check for completed experiments and free up resources ---
        for port, info in list(running_processes.items()):
            return_code = info['process'].poll()
            if return_code is not None:
                gpu_id = info['gpu']
                config = info['config']
                
                print(f"\n--- Experiment Finished on Port {port} (GPU {gpu_id}) ---")
                print(f"Model: {config['model']}, Dataset: {config['dataset']}, Optimizer: {config['optim']}")
                print(f"Memory per trial: {config.get('memory_per_trial', 'N/A')} MB")
                print(f"Status: {'SUCCESS' if return_code == 0 else f'FAILED (Exit Code: {return_code})'}")
                print("----------------------------------------")

                info['log_file'].close()
                del running_processes[port]
                gpu_running_count[gpu_id] -= 1
                gpu_memory_used[gpu_id] -= estimate_experiment_memory_usage(config)

                pbar.update(1)
                pbar.set_postfix_str(f"Running: {len(running_processes)}")

                time.sleep(5)

        # --- 2. Launch new experiments if resources are available ---
        # --- START: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
        # Try to launch experiments on each GPU that has available capacity
        for gpu_id in args.gpus:
            # Check if this GPU has experiments in queue and capacity to run more
            if (gpu_queues[gpu_id] and 
                gpu_running_count[gpu_id] < args.max_experiments_per_gpu and
                gpu_memory_used[gpu_id] + estimate_experiment_memory_usage(gpu_queues[gpu_id][0]) <= args.max_memory_per_gpu):
                
                config_to_run = gpu_queues[gpu_id].popleft()
                max_trials_per_gpu = config_to_run['optimal_trials_per_gpu']
                
                port_to_use = find_available_port(next_port)
                next_port = port_to_use + 1

                print(f"\n--- Launching Experiment on Port {port_to_use} (GPU {gpu_id}) ---")
                print(f"Model: {config_to_run['model']}, Dataset: {config_to_run['dataset']}, Optimizer: {config_to_run['optim']}")
                print(f"Memory per trial: {config_to_run.get('memory_per_trial', 'N/A')} MB")
                print(f"Assigned GPU: {gpu_id}, Optimal Trials per GPU: {max_trials_per_gpu}")
                print(f"GPU {gpu_id} running: {gpu_running_count[gpu_id] + 1} experiments")
                print(f"Experiments remaining on GPU {gpu_id}: {len(gpu_queues[gpu_id])}")
                print("----------------------------------------")
                
                cmd = [
                    sys.executable, "run_nni.py",
                    "--model", config_to_run['model'],
                    "--dataset", config_to_run['dataset'],
                    "--optim", config_to_run['optim'],
                    "--fold", str(config_to_run['fold']),
                    "--num_epochs", str(config_to_run['num_epochs']),
                    "--port", str(port_to_use),
                    "--gpu_index", str(gpu_id),
                    "--trial_concurrency", str(args.trial_concurrency),
                    "--max_trials_per_gpu", str(max_trials_per_gpu),
                ]
                if config_to_run.get('norm', False):
                    cmd.append("--norm")
                if 'num_layers' in config_to_run:
                    cmd.extend(["--num_layers", str(config_to_run['num_layers'])])
                
                try:
                    log_dir = "experiment_logs"
                    os.makedirs(log_dir, exist_ok=True)
                    log_filename = f"{log_dir}/{config_to_run['dataset']}_{config_to_run['model']}_{config_to_run['optim']}.log"
                    log_file = open(log_filename, 'w')

                    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                    running_processes[port_to_use] = {
                        "process": process,
                        "gpu": gpu_id,
                        "config": config_to_run,
                        "log_file": log_file
                    }
                    
                    gpu_running_count[gpu_id] += 1
                    gpu_memory_used[gpu_id] += estimate_experiment_memory_usage(config_to_run)

                    time.sleep(5)
                    pbar.set_postfix_str(f"Running: {len(running_processes)}")
                except Exception as e:
                    print(f"!!! ERROR: Failed to launch experiment for config: {config_to_run}. Error: {e}")
                    # Put the config back in the queue if launch fails
                    gpu_queues[gpu_id].appendleft(config_to_run)
        # --- END: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---

        if not any(gpu_queues.values()) and not running_processes:
            break
            
        # Calculate overall progress
        remaining_experiments = sum(len(queue) for queue in gpu_queues.values())
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {len(running_processes)} running, {remaining_experiments} pending. "
              f"GPU loads: {dict(gpu_running_count)}. Waiting for {args.check_interval}s...")
        time.sleep(args.check_interval)
    
    print("\n--- All experiments have been completed. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Aware Packing NNI Experiment Scheduler")
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='Comma-separated list of GPU indices to use (e.g., "0,1,2,3").')
    # --- START: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
    parser.add_argument('--max_memory_per_gpu', type=int, required=True, help='Maximum available memory per GPU in MB.')
    parser.add_argument('--max_experiments_per_gpu', type=int, default=10, help='Maximum number of experiments to run concurrently on a single GPU.')
    # --- END: MODIFICATIONS FOR MEMORY-AWARE PACKING SCHEDULING ---
    parser.add_argument('--trial_concurrency', type=int, default=128, help='Number of concurrent trials for each experiment.')
    parser.add_argument('--max_trials_per_gpu', type=int, default=32, help='Default maximum number of concurrent trials per GPU (overridden by memory-aware calculation).')
    parser.add_argument('--start_port', type=int, default=50000, help='The starting port number for NNI experiments.')
    parser.add_argument('--check_interval', type=int, default=60, help='Seconds to wait between checking experiment statuses.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)

'''
python scheduler_plus.py \
--gpus 0,1,2,3 \
--max_memory_per_gpu 16384 \
--max_experiments_per_gpu 8 \
--trial_concurrency 128 \
--start_port 50000 \
--check_interval 120
'''
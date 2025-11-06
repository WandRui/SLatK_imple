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

# --- START: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---
def load_memory_config(memory_file="memory_test_results/memory_test.json"):
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
    """Generates all experiment configurations sorted by memory usage (descending for better packing)."""
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
    
    # Sort by memory usage (descending) - run memory-intensive experiments first for better packing
    configs.sort(key=lambda x: x['memory_per_trial'], reverse=True)
    return configs

def estimate_experiment_memory_usage(config):
    """Estimate total memory usage for an experiment."""
    # Estimate based on memory per trial and number of concurrent trials
    memory_per_trial = config.get('memory_per_trial', 1000)
    optimal_trials = config.get('optimal_trials_per_gpu', 8)
    
    # Add some overhead for the experiment itself
    experiment_overhead = 100  # MB
    return memory_per_trial * optimal_trials + experiment_overhead

def find_best_gpu_for_experiment(config, gpu_memory_available, gpu_running_count, max_experiments_per_gpu):
    """Find the best GPU to run the experiment based on current memory availability."""
    config_memory = estimate_experiment_memory_usage(config)
    
    # Find GPUs that have capacity for this experiment
    suitable_gpus = []
    for gpu_id, available_memory in gpu_memory_available.items():
        if (available_memory >= config_memory and 
            gpu_running_count[gpu_id] < max_experiments_per_gpu):
            suitable_gpus.append((gpu_id, available_memory))
    
    if not suitable_gpus:
        return None  # No GPU has enough memory or capacity
    
    # Prefer GPUs with the least free memory (best-fit strategy to reduce fragmentation)
    suitable_gpus.sort(key=lambda x: x[1])  # Sort by available memory (ascending)
    return suitable_gpus[0][0]  # Return GPU with smallest sufficient space

def get_next_experiment_for_gpu(gpu_id, remaining_configs, gpu_memory_available, max_experiments_per_gpu):
    """Find the best experiment to run on a specific GPU."""
    available_memory = gpu_memory_available[gpu_id]
    
    # Try to find an experiment that fits well in the available memory
    best_config = None
    best_memory_usage = 0
    
    for config in remaining_configs:
        config_memory = estimate_experiment_memory_usage(config)
        
        # Check if this config fits and is better than current best
        if config_memory <= available_memory:
            if best_config is None or config_memory > best_memory_usage:
                best_config = config
                best_memory_usage = config_memory
    
    return best_config
# --- END: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---

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

    # --- START: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---
    # Load memory configuration and calculate optimal trials
    memory_config = load_memory_config()
    memory_optimal_configs = calculate_optimal_trials_per_gpu(memory_config, args.max_memory_per_gpu)
    
    # Generate configurations sorted by memory usage (descending)
    all_configs = get_sorted_experiment_configs(memory_optimal_configs)
    remaining_configs = deque(all_configs)  # Use deque for efficient popping
    
    # Initialize GPU state
    gpu_memory_available = {gpu: args.max_memory_per_gpu for gpu in args.gpus}
    gpu_running_count = {gpu: 0 for gpu in args.gpus}
    
    # Calculate total experiments
    total_experiments = len(all_configs)
    
    print(f"--- Dynamic Memory-Aware Scheduler Initialized ---")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Available GPUs: {sorted(list(args.gpus))}")
    print(f"Maximum GPU memory per GPU: {args.max_memory_per_gpu} MB")
    print(f"Maximum experiments per GPU: {args.max_experiments_per_gpu}")
    print("Experiment execution order (by memory usage, descending):")
    for i, config in enumerate(all_configs[:5]):  # Show first 5
        print(f"  {i+1}. {config['model']} + {config['optim']} on {config['dataset']}: {config['memory_per_trial']} MB/trial")
    if len(all_configs) > 5:
        print(f"  ... and {len(all_configs) - 5} more")
    print("----------------------------------------")
    # --- END: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---
    
    next_port = args.start_port
    pbar = tqdm(total=total_experiments, desc="Overall Progress", unit="exp")
    
    while remaining_configs or running_processes:
        
        # --- 1. Check for completed experiments and free up resources ---
        for port, info in list(running_processes.items()):
            return_code = info['process'].poll()
            if return_code is not None:
                gpu_id = info['gpu']
                config = info['config']
                memory_freed = estimate_experiment_memory_usage(config)
                
                print(f"\n--- Experiment Finished on Port {port} (GPU {gpu_id}) ---")
                print(f"Model: {config['model']}, Dataset: {config['dataset']}, Optimizer: {config['optim']}")
                print(f"Memory per trial: {config.get('memory_per_trial', 'N/A')} MB")
                print(f"Freed memory: {memory_freed} MB")
                print(f"Status: {'SUCCESS' if return_code == 0 else f'FAILED (Exit Code: {return_code})'}")
                print("----------------------------------------")

                info['log_file'].close()
                del running_processes[port]
                gpu_running_count[gpu_id] -= 1
                gpu_memory_available[gpu_id] += memory_freed

                pbar.update(1)
                pbar.set_postfix_str(f"Running: {len(running_processes)}, Remaining: {len(remaining_configs)}")

                time.sleep(5)  # Brief pause after completion

        # --- 2. Launch new experiments if resources are available ---
        # --- START: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---
        if remaining_configs:
            # Try to find suitable experiments for each available GPU
            experiments_launched = 0
            
            for gpu_id in args.gpus:
                # Skip if GPU is at capacity
                if gpu_running_count[gpu_id] >= args.max_experiments_per_gpu:
                    continue
                
                # Find the best experiment for this GPU
                config_to_run = get_next_experiment_for_gpu(
                    gpu_id, remaining_configs, gpu_memory_available, args.max_experiments_per_gpu
                )
                
                if config_to_run:
                    # Remove the config from remaining list
                    remaining_configs.remove(config_to_run)
                    
                    port_to_use = find_available_port(next_port)
                    next_port = port_to_use + 1
                    max_trials_per_gpu = config_to_run['optimal_trials_per_gpu']
                    config_memory = estimate_experiment_memory_usage(config_to_run)
                    
                    print(f"\n--- Launching Experiment on Port {port_to_use} (GPU {gpu_id}) ---")
                    print(f"Model: {config_to_run['model']}, Dataset: {config_to_run['dataset']}, Optimizer: {config_to_run['optim']}")
                    print(f"Memory per trial: {config_to_run.get('memory_per_trial', 'N/A')} MB")
                    print(f"Total experiment memory: {config_memory} MB")
                    print(f"Assigned GPU: {gpu_id}, Optimal Trials per GPU: {max_trials_per_gpu}")
                    print(f"GPU {gpu_id} running: {gpu_running_count[gpu_id] + 1}/{args.max_experiments_per_gpu} experiments")
                    print(f"GPU {gpu_id} memory: {gpu_memory_available[gpu_id] - config_memory:.0f}/{args.max_memory_per_gpu} MB remaining")
                    print(f"Experiments remaining: {len(remaining_configs)}")
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
                        gpu_memory_available[gpu_id] -= config_memory
                        experiments_launched += 1

                        time.sleep(5)  # Brief pause between launches
                        pbar.set_postfix_str(f"Running: {len(running_processes)}, Remaining: {len(remaining_configs)}")
                    except Exception as e:
                        print(f"!!! ERROR: Failed to launch experiment for config: {config_to_run}. Error: {e}")
                        # Put the config back in the queue if launch fails
                        remaining_configs.appendleft(config_to_run)
            
            # If no experiments could be launched but some are remaining, wait longer
            if experiments_launched == 0 and remaining_configs:
                # Show why we can't launch experiments
                print(f"\n[DEBUG] Cannot launch experiments due to resource constraints:")
                for gpu_id in args.gpus:
                    available_mem = gpu_memory_available[gpu_id]
                    running_count = gpu_running_count[gpu_id]
                    next_exp_mem = estimate_experiment_memory_usage(remaining_configs[0]) if remaining_configs else 0
                    print(f"  GPU {gpu_id}: {running_count}/{args.max_experiments_per_gpu} experiments, "
                          f"{available_mem:.0f}MB available, next experiment needs {next_exp_mem:.0f}MB")
        # --- END: MODIFICATIONS FOR DYNAMIC MEMORY-AWARE SCHEDULING ---

        if not remaining_configs and not running_processes:
            break
            
        # Calculate overall progress
        remaining_count = len(remaining_configs)
        running_count = len(running_processes)
        
        # Display detailed GPU status
        status_lines = []
        for gpu_id in sorted(args.gpus):
            running = gpu_running_count[gpu_id]
            available_mem = gpu_memory_available[gpu_id]
            status_lines.append(f"GPU{gpu_id}: {running} running, {available_mem:.0f}MB free")
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {running_count} running, {remaining_count} pending. "
              f"[{', '.join(status_lines)}]. Waiting for {args.check_interval}s...")
        time.sleep(args.check_interval)
    
    print("\n--- All experiments have been completed. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Memory-Aware NNI Experiment Scheduler")
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='Comma-separated list of GPU indices to use (e.g., "0,1,2,3").')
    parser.add_argument('--max_memory_per_gpu', type=int, required=True, help='Maximum available memory per GPU in MB.')
    parser.add_argument('--max_experiments_per_gpu', type=int, default=4, help='Maximum number of experiments to run concurrently on a single GPU.')
    parser.add_argument('--trial_concurrency', type=int, default=128, help='Number of concurrent trials for each experiment.')
    parser.add_argument('--max_trials_per_gpu', type=int, default=32, help='Default maximum number of concurrent trials per GPU (overridden by memory-aware calculation).')
    parser.add_argument('--start_port', type=int, default=50000, help='The starting port number for NNI experiments.')
    parser.add_argument('--check_interval', type=int, default=60, help='Seconds to wait between checking experiment statuses.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)

'''
python -u scheduler_plus.py \
  --gpus 4,5,6,7 \
  --max_memory_per_gpu 23000 \
  --max_experiments_per_gpu 4 \
  --trial_concurrency 64 \
  --start_port 40000 \
  --check_interval 180 > experiment_logs/scheduler_plus.log
'''
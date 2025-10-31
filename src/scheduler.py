import os
import sys
import time
import subprocess
import argparse
import signal
from collections import deque
from typing import List, Dict, Any
from nni.experiment import Experiment
from tqdm import tqdm
from itemrec.port_handle import find_available_port


# --- Global state for signal handling ---
running_processes: Dict[int, Dict[str, Any]] = {}

# Model-specific maximum trials per GPU
MODEL_SPECIFIC_CONFIG = {
    'MF': {
        'max_trials_per_gpu': 32  # MF is lightweight, can run many trials.
    },
    'LightGCN': {
        'max_trials_per_gpu': 16  # LightGCN is moderately heavy.
    },
    'XSimGCL': {
        'max_trials_per_gpu': 8   # XSimGCL is memory-intensive.
    }
    # Add other models here if they have different requirements.
}

def get_experiment_configs() -> List[Dict[str, Any]]:
    """Generates all experiment configurations."""
    MODELS = ("MF", "LightGCN", "XSimGCL")
    LOSSES = ("BPR", "GuidedRec", "LLPAUC", "Softmax", "AdvInfoNCE", "BSL", "PSL", "SLatK")
    DATASETS = ("amazon2014-health", "amazon2014-electronic", "amazon2014-book", "gowalla")
    
    configs = []
    for dataset in DATASETS:
        for optim in LOSSES:
            for model in MODELS:
                config = {
                    "model": model,
                    "dataset": dataset,
                    "optim": optim,
                    "norm": True,
                    "fold": 1,
                    "num_epochs": 200,
                }
                configs.append(config)
    return configs

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

    configs_to_run = deque(get_experiment_configs())
    # --- START: MODIFICATIONS FOR SCHEDULER ---
    # Instead of a set of available GPUs, use a dictionary to track the load on each GPU.
    gpu_loads = {gpu_id: 0 for gpu_id in args.gpus}
    # The overall concurrency is now simply what the user specifies.
    max_concurrency = args.concurrency
    # --- END: MODIFICATIONS FOR SCHEDULER ---
    next_port = args.start_port

    total_experiments = len(configs_to_run)
    pbar = tqdm(total=total_experiments, desc="Overall Progress", unit="exp")
    
    print(f"--- Experiment Scheduler Initialized ---")
    print(f"Total experiments to run: {len(configs_to_run)}")
    print(f"Available GPUs: {sorted(list(gpu_loads.keys()))}")
    print(f"Maximum concurrent experiments: {max_concurrency}")
    print("----------------------------------------")

    while configs_to_run or running_processes:
        
        # --- 1. Check for completed experiments and free up resources ---
        for port, info in list(running_processes.items()):
            return_code = info['process'].poll()
            if return_code is not None:
                gpu_id = info['gpu']
                config = info['config']
                
                print(f"\n--- Experiment Finished on Port {port} (GPU {gpu_id}) ---")
                print(f"Model: {config['model']}, Dataset: {config['dataset']}, Optimizer: {config['optim']}")
                print(f"Status: {'SUCCESS' if return_code == 0 else f'FAILED (Exit Code: {return_code})'}")
                print("----------------------------------------")

                info['log_file'].close()
                del running_processes[port]
                # --- START: MODIFICATIONS FOR SCHEDULER ---
                # Decrement the load count for the GPU that has been freed up.
                gpu_loads[gpu_id] -= 1
                # --- END: MODIFICATIONS FOR SCHEDULER ---

                pbar.update(1)
                pbar.set_postfix_str(f"Running: {len(running_processes)}")

                time.sleep(5)

        # --- 2. Launch new experiments if resources are available ---
        # --- START: MODIFICATIONS FOR SCHEDULER ---
        # The condition no longer checks for "available_gpus" but just for overall concurrency.
        while len(running_processes) < max_concurrency and configs_to_run:
            # Find the GPU with the minimum current load to assign the next job.
            gpu_to_use = min(gpu_loads, key=gpu_loads.get)
            gpu_loads[gpu_to_use] += 1  # Increment the load for the chosen GPU.
            # --- END: MODIFICATIONS FOR SCHEDULER ---
            
            config_to_run = configs_to_run.popleft()
            max_trials_per_gpu = MODEL_SPECIFIC_CONFIG.get(config_to_run['model'], {}).get('max_trials_per_gpu', args.max_trials_per_gpu)
            port_to_use = find_available_port(next_port)
            next_port = port_to_use + 1

            print(f"\n--- Launching Experiment on Port {port_to_use} (GPU {gpu_to_use}) ---")
            print(f"Model: {config_to_run['model']}, Dataset: {config_to_run['dataset']}, Optimizer: {config_to_run['optim']}")
            print(f"Assigned GPU: {gpu_to_use}, Max Trials per GPU: {max_trials_per_gpu}")
            print(f"Current GPU Loads: {gpu_loads}")
            print(f"Experiments remaining: {len(configs_to_run)}")
            print("----------------------------------------")
            
            cmd = [
                sys.executable, "run_nni.py",
                "--model", config_to_run['model'],
                "--dataset", config_to_run['dataset'],
                "--optim", config_to_run['optim'],
                "--fold", str(config_to_run['fold']),
                "--num_epochs", str(config_to_run['num_epochs']),
                "--port", str(port_to_use),
                "--gpu_index", str(gpu_to_use),
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
                    "gpu": gpu_to_use,
                    "config": config_to_run,
                    "log_file": log_file
                }

                time.sleep(5)
                pbar.set_postfix_str(f"Running: {len(running_processes)}")
            except Exception as e:
                print(f"!!! ERROR: Failed to launch experiment for config: {config_to_run}. Error: {e}")
                gpu_loads[gpu_to_use] -= 1 # Decrement load if launch fails

        if not (configs_to_run or running_processes):
            break
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status check: {len(running_processes)} running, {len(configs_to_run)} pending. GPUs Loads: {gpu_loads}. Waiting for {args.check_interval}s...")
        time.sleep(args.check_interval)
    
    print("\n--- All experiments have been completed. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NNI Experiment Scheduler")
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='Comma-separated list of GPU indices to use (e.g., "0,1,2,3").')
    parser.add_argument('--concurrency', type=int, default=8, help='Maximum number of experiments to run in parallel.')
    parser.add_argument('--trial_concurrency', type=int, default=128, help='Number of concurrent trials for each experiment.')
    parser.add_argument('--max_trials_per_gpu', type=int, default=32, help='Maximum number of concurrent trials for each experiment on its assigned GPU. (This is overridden by model-specific settings if applicable.)')
    parser.add_argument('--start_port', type=int, default=50000, help='The starting port number for NNI experiments.')
    parser.add_argument('--check_interval', type=int, default=60, help='Seconds to wait between checking experiment statuses.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)
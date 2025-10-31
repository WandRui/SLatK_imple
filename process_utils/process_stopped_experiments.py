#!/usr/bin/env python3
"""
Script to process NNI experiments with STOPPED status.
For each stopped experiment, it will:
1. Run 'nnictl view [experiment_id]' and wait 10 seconds
2. Run 'nnictl experiment export [experiment_id] --filename experiment_results/[experimentName] --type json' and wait 2 seconds
3. Run 'nnictl experiment stop [experiment_id]'
"""

import json
import subprocess
import time
import os
from pathlib import Path

def load_experiments(experiment_file_path):
    """Load experiments from the .experiment file."""
    try:
        with open(experiment_file_path, 'r') as f:
            experiments = json.load(f)
        return experiments
    except FileNotFoundError:
        print(f"Error: {experiment_file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def run_command(command, wait_time=0):
    """Run a shell command and optionally wait."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Command executed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"✗ Command failed with return code {result.returncode}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
        
        if wait_time > 0:
            print(f"Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Exception running command: {e}")
        return False

def ensure_results_directory():
    """Ensure the experiment_results directory exists."""
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    print(f"✓ Results directory ensured: {results_dir.absolute()}")

def process_stopped_experiments():
    """Main function to process all stopped experiments."""
    # Get the script directory and construct path to .experiment file
    script_dir = Path(__file__).parent
    experiment_file = script_dir / "nni-experiments" / ".experiment"
    
    print(f"Loading experiments from: {experiment_file.absolute()}")
    
    # Load experiments
    experiments = load_experiments(experiment_file)
    if experiments is None:
        return
    
    # Ensure results directory exists
    ensure_results_directory()

    # list all json files in experiment_results directory
    existing_experiments = list(Path("experiment_results").glob("*.json"))
    existing_experiments = [path.replace(".json", "").replace("experiment_results/", "") for path in map(str, existing_experiments)]
    # print(f"Existing processed experiments: {existing_experiments}")
    
    # Filter stopped experiments, excluding those already processed
    stopped_experiments = {
        exp_id: exp_data for exp_id, exp_data in experiments.items() 
        if exp_data.get('status') == 'STOPPED' # and f"{exp_data.get('experimentName', '')}" not in existing_experiments
    }

    # print(f"\nFound {len(stopped_experiments)} stopped and unprocessed experiments out of {len(experiments)} total experiments")
    # print(f"experiments to process: {list(stopped_experiments.keys())}")

    if not stopped_experiments:
        print("No stopped experiments to process.")
        return
    
    # Process each stopped experiment
    for i, (exp_id, exp_data) in enumerate(stopped_experiments.items(), 1):
        if not exp_data:
            print(f"No Experiment data for ID {exp_id} is missing or invalid. Skipping.")
            continue
        experiment_name = exp_data.get('experimentName', f'experiment_{exp_id}')
        
        print(f"\n{'='*60}")
        print(f"Processing experiment {i}/{len(stopped_experiments)}")
        print(f"ID: {exp_id}")
        print(f"Name: {experiment_name}")
        print(f"{'='*60}")
        
        # Step 1: View experiment (wait 2 seconds)
        print(f"\nStep 1: Viewing experiment {exp_id}")
        run_command(f"nnictl view {exp_id} --port {exp_data.get('port', 8080)}", wait_time=2)
        
        # Step 2: Export experiment (wait 2 seconds)
        print(f"\nStep 2: Exporting experiment {exp_id}")
        export_filename = f"experiment_results/{experiment_name}.json"
        run_command(f"nnictl experiment export {exp_id} --filename {export_filename} --type json", wait_time=2)
        
        # Step 3: Stop experiment
        print(f"\nStep 3: Stopping experiment {exp_id}")
        run_command(f"nnictl stop {exp_id}")
        
        print(f"✓ Completed processing experiment {exp_id}")
    
    print(f"\n{'='*60}")
    print(f"✓ All {len(stopped_experiments)} stopped experiments have been processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("NNI Stopped Experiments Processor")
    print("=" * 40)
    
    # Change to the script directory to ensure relative paths work correctly
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        process_stopped_experiments()
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
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

def is_json_file_empty_list(file_path):
    """Check if a JSON file contains an empty list."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) == 0
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        # If file doesn't exist or can't be parsed, consider it as needing reprocessing
        return True

def process_stopped_experiments():
    """Main function to process all stopped experiments."""
    # Get the script directory and construct path to .experiment file
    cur_dir = Path(os.getcwd())
    experiment_file = cur_dir / "nni-experiments" / ".experiment"

    print(f"Loading experiments from: {experiment_file.absolute()}")
    
    # Load experiments
    experiments = load_experiments(experiment_file)
    if experiments is None:
        return
    
    # Ensure results directory exists
    ensure_results_directory()

    # list all json files in experiment_results directory
    existing_experiment_files = list(Path("experiment_results").glob("*.json"))
    
    # Check which experiments are valid (not empty lists)
    valid_experiments = set()
    empty_experiments = set()
    
    for json_file in existing_experiment_files:
        experiment_name = json_file.stem  # Get filename without extension
        if is_json_file_empty_list(json_file):
            empty_experiments.add(experiment_name)
            print(f"Found empty experiment file: {json_file}")
        else:
            valid_experiments.add(experiment_name)
    
    print(f"Valid experiments: {len(valid_experiments)}, Empty experiments: {len(empty_experiments)}")
    
    # Filter stopped experiments - process if not in valid_experiments (include empty ones for reprocessing)
    stopped_experiments = {
        exp_id: exp_data for exp_id, exp_data in experiments.items()
        if exp_data.get('status') == 'STOPPED' and exp_data.get('experimentName') not in valid_experiments
    }

    # Show which experiments will be reprocessed due to empty results
    reprocessing_experiments = []
    for exp_id, exp_data in stopped_experiments.items():
        if exp_data.get('experimentName') in empty_experiments:
            reprocessing_experiments.append(f"{exp_id} ({exp_data.get('experimentName')})")
    
    if reprocessing_experiments:
        print(f"\nExperiments to reprocess due to empty results: {reprocessing_experiments}")
    
    print(f"\nFound {len(stopped_experiments)} stopped experiments to process out of {len(experiments)} total experiments")
    # Find name of stopped experiments
    for exp_id, exp_data in stopped_experiments.items():
        print(f" - ID: {exp_id}, Name: {exp_data.get('experimentName', 'N/A')}")
    

    # exit(0) # Comment this line to enable processing
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
        run_command(f"nnictl view {exp_id} --port {exp_data.get('port', 8080)}", wait_time=8)
        
        # Step 2: Export experiment (wait 2 seconds)
        print(f"\nStep 2: Exporting experiment {exp_id}")
        export_filename = f"experiment_results/{experiment_name}.json"
        export_success = run_command(f"nnictl experiment export {exp_id} --filename {export_filename} --type json", wait_time=5)
        
        # Check if exported file is empty
        if export_success and Path(export_filename).exists():
            if is_json_file_empty_list(export_filename):
                print(f"⚠️  WARNING: Exported file {export_filename} contains an empty list - experiment may have failed!")
            else:
                print(f"✓ Successfully exported non-empty results to {export_filename}")
        
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
    print(f"Working directory: {os.getcwd()}")
    
    try:
        process_stopped_experiments()
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: NNI Main Entry
# Description:
#  This module is not a part of the ItemRec project, but it is used
#  to run the ItemRec project with NNI (Neural Network Intelligence).
#  If you want to run the ItemRec with NNI, you can directly run this
#  module.
#  NOTE: You can modify this code to fit your own project.
# -------------------------------------------------------------------

# import modules ----------------------------------------------------
from typing import (
    Any, 
    Optional,
    List,
    Tuple,
    Set,
    Dict,
    Callable,
)
from itemrec.hyper import get_search_space
import argparse
import nni
import sys
from nni.experiment import Experiment
from itemrec.port_handle import is_port_available
import time

# public functions --------------------------------------------------
__all__ = [

]

# arguments ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    r"""
    Parse the arguments for the main function.
    """
    parser = argparse.ArgumentParser(description='ItemRec with NNI')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name of the model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The name of the dataset'
    )
    parser.add_argument(
        '--optim',
        type=str,
        required=True,
        help='The name of the optimizer'
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalize the embeddings. If true, the cosine similarity is used in the evaluation.",
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Whether to use the out-of-distribution test set."
    )
    parser.add_argument(
        "--neg_num", 
        type=int,
        default=1000,
        help="The number of negative samples"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='The number of epochs'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='The number of layers of the LightGCN model'
    )
    parser.add_argument(
        '--contrast_weight', 
        type=float,
        default=0.2,
        help='The weight of the contrastive loss in XSimGCL'
    )
    parser.add_argument(
        '--method',
        type=int,
        default=1,
        help='The method of the PSL optimizer'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='tanh',
        help='The activation function of the PSL optimizer'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='The value of k in SLatK'
    )
    parser.add_argument(
        '--no_valid',
        action='store_true',
        help='Whether to use the validation set'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='uniform',
        help='The sampler for negative sampling'
    )
    parser.add_argument(
        '--epoch_sampler',
        type=int,
        default=0,
        help='The number of epochs to update the sampler'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        help='The number of folds for cross-validation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='The number of workers for the data loader'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=10000,
        help='The port of the NNI experiment'
    )
    # --- START: MODIFICATIONS FOR SCHEDULER ---
    parser.add_argument(
        '--gpu_index',
        type=int,
        required=True,
        help='The index of the GPU to use for this experiment.'
    )
    parser.add_argument(
        '--max_trials_per_gpu',
        type=int,
        default=32,
        help='The maximum number of concurrent trials to run on the assigned GPU.'
    )
    parser.add_argument(
        '--trial_concurrency',
        type=int,
        default=128,
        help='The number of concurrent trials for this experiment.'
    )
    # --- END: MODIFICATIONS FOR SCHEDULER ---
    return parser.parse_args()

def get_total_trials(space: Dict[str, Any]) -> int:
        total = 1
        for param, config in space.items():
            if config['_type'] == 'choice':
                total *= len(config['_value'])
            elif config['_type'] == 'quniform':
                low, high, q = config['_value']
                total *= int((high - low) / q) + 1
            else:
                raise ValueError(f"Unsupported hyperparameter type: {config['_type']}")
        return total

# main function -----------------------------------------------------
def main():
    args = parse_args()

    # TODO: `/path/to/your/` must be replaced with the actual paths
    save_dir = f"memory_test_save/{args.dataset}/{args.model}/{args.optim}"
    if not args.ood:
        dataset_path = f"IR-Benchmark-Dataset/data_iid/{args.dataset}"
    else:
        dataset_path = f"IR-Benchmark-Dataset/data_ood/{args.dataset}"
    norm_cmd = "--norm " if args.norm else ""
    if args.model == 'LightGCN':
        model_cmd = f"LightGCN --num_layers={args.num_layers} "
    elif args.model == 'LightGCNPP':
        model_cmd = f"LightGCNPP --num_layers={args.num_layers} "
    elif args.model == 'MF':
        model_cmd = f"MF "
    elif args.model == 'NCF':
        model_cmd = f"NCF "
    elif args.model == 'SimGCL':
        model_cmd = f"SimGCL --contrast_weight={args.contrast_weight} "
    elif args.model == 'SimpleX':
        model_cmd = f"SimpleX --history_len=50 --history_weight=0.5 "
    elif args.model == 'XSimGCL':
        model_cmd = f"XSimGCL --contrast_weight={args.contrast_weight} "
    else:
        raise ValueError(f"Invalid model: {args.model}")
    if args.optim == 'AdvInfoNCE':
        optim_cmd = f"AdvInfoNCE --neg_num={args.neg_num} --tau=0.2 --neg_weight=64 --lr_adv=5e-5 --epoch_adv=5 "
    elif args.optim == 'BPR':
        optim_cmd = f"BPR "
    elif args.optim == 'BSL':
        optim_cmd = f"BSL --neg_num={args.neg_num} --tau1=0.2 --tau2=0.2 "
    elif args.optim == 'GuidedRec':
        optim_cmd = f"GuidedRec --neg_num=9 "
    elif args.optim == 'LambdaRank':
        optim_cmd = f"LambdaRank "
    elif args.optim == 'LambdaLoss':
        optim_cmd = f"LambdaLoss "
    elif args.optim == 'LambdaLossAtK':
        optim_cmd = f"LambdaLossAtK --k={args.k} "
    elif args.optim == 'LLPAUC':
        optim_cmd = f"LLPAUC --neg_num={args.neg_num} --alpha=0.7 --beta=0.1 "
    elif args.optim == 'PSL':
        assert args.method in [1, 2]
        assert args.activation in ['tanh', 'relu', 'atan']
        optim_cmd = f"PSL --neg_num={args.neg_num} --tau=2.0 --tau_star=0.1 --method={args.method} --activation={args.activation} "
    elif args.optim == 'SLatK':
        optim_cmd = f"SLatK --neg_num={args.neg_num} --tau=0.2 --tau_beta=1.0 --k={args.k} --epoch_quantile=5 "
    elif args.optim == 'Softmax':
        optim_cmd = f"Softmax --neg_num={args.neg_num} --tau=0.2 "
    elif args.optim == 'SogCLR':
        optim_cmd = f"SogCLR --neg_num={args.neg_num} --tau=0.2 --gamma_g=0.9 "
    elif args.optim == 'SogSLatK':
        optim_cmd = f"SogSLatK --neg_num={args.neg_num} --tau=0.2 --tau_beta=1.0 --k={args.k} --epoch_quantile=5 --gamma_g=0.9 "
    else:
        raise ValueError(f"Invalid optimizer: {args.optim}")

    # ItemRec command
    cmd = f"python -u -m itemrec " \
        f"--log={save_dir}/ir.log " \
        f"--save_dir={save_dir} " \
        f"--seed=2024 " \
        f"model --emb_size=64 " + norm_cmd + f"--num_epochs={args.num_epochs} " + model_cmd + \
        f"dataset --data_path={dataset_path} --batch_size=1024 --num_workers={args.num_workers} " \
            + ("--no_valid " if args.ood or args.no_valid else "") \
            + f"--sampler={args.sampler} --epoch_sampler={args.epoch_sampler} --fold={args.fold} " + \
        f"optim --lr=1e-1 --weight_decay=0.0 " + optim_cmd
        # f"--truncate 1024 "
    
    # NNI experiment
    experiment = Experiment('local')
    experiment.config.experiment_name = f'ItemRec-{args.dataset}-{args.model}-{args.optim}'

    experiment.config.trial_command = cmd
    experiment.config.trial_code_directory = '.'
    experiment.config.trial_concurrency = args.trial_concurrency
    experiment.config.trial_gpu_number = 1
    
    # --- START: MODIFICATIONS FOR SCHEDULER ---
    # Total concurrency for this single experiment
    
    experiment.config.training_service.max_trial_number_per_gpu = args.max_trials_per_gpu
    # Assign the experiment to one specific GPU
    experiment.config.training_service.gpu_indices = [args.gpu_index]
    # --- END: MODIFICATIONS FOR SCHEDULER ---
    experiment.config.training_service.platform = 'local'
    experiment.config.training_service.use_active_gpu = True

    # This setting is now controlled by the scheduler launching this script
    # experiment.config.training_service.max_trial_number_per_gpu = 32
    search_space = get_search_space(args.optim)
    experiment.config.search_space = search_space

    total_trial_number = get_total_trials(search_space)

    experiment.config.max_trial_number = total_trial_number
    # experiment.config.max_exec_duration = '2h'

    # Grid Search has no optimize mode
    experiment.config.tuner.name = 'GridSearch'
    
    # experiment.config.tuner.name = 'TPE'
    # experiment.config.tuner.class_args = {
    #     'optimize_mode': 'maximize'
    # }

    # Check if the port is available
    if is_port_available(args.port) is False:
        print(f"Port {args.port} is not available. Please choose another port.")
        sys.exit(1)

    try:
        success = experiment.run(port=args.port, wait_completion=True, debug=True)
        print(f"NNI experiment started on port {args.port} with ID: {experiment.id}")
        status = experiment.get_status()
        print(f"Experiment {experiment.id} final status: {status}")
        # Check experiment result
        if status == 'DONE':
            print("NNI experiment completed successfully.")
            experiment.stop()
            sys.exit(0)
        else:
            print("NNI experiment failed.")
            experiment.stop()
            sys.exit(1)
    except Exception as e:
        print(f"An exception occurred while running the experiment on port {args.port}: {e}")
        experiment.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
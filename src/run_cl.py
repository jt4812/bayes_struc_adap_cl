import argparse
import json
import os
import time

import torch

from src.utils.cl_train import train
from src.utils.utils import update_config

HYPER_PARAMS_FOLDER = "hyper_params"

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train', add_help=True)
    parser.add_argument('--seed', type=int, default=None, help='seed to initialize the experiment')
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=["split_mnist", "split_fashion_mnist", "perm_mnist", "split_cifar10", "split_cifar100_10",
                 "split_cifar100_20", "split_tiny_imagenet_10"],
        help='Dataset to experiment on.')
    parser.add_argument('--n_tasks', type=int, default=None, help='Number of tasks')
    parser.add_argument('--single_head', action="store_true", help='Whether to use single head or multi head')
    parser.add_argument('--mask_type', type=str, default="weight_mask", choices=["weight_mask", "neuron_mask"],
                        help='Use fixed mask for each task')
    parser.add_argument('--architecture', type=str, default="alexnet", help='Model Architecture')
    parser.add_argument('--width', default=200, type=int, help='model wdith')

    parser.add_argument('--lr_weight', default=None, type=float, help='lr weight')
    parser.add_argument('--lr_struc', default=None, type=float, help='lr struc')
    parser.add_argument('--decay_rate_weight', default=None, type=float, help='lr weight decay rate')
    parser.add_argument('--n_epochs', default=None, type=int, help='n epochs')
    parser.add_argument('--n_epochs_per_finetune_task', default=None, type=int, help='n_epochs_per_task_finetune')

    parser.add_argument('--alpha', default=None, type=float, help='alpha')
    parser.add_argument('--beta', default=None, type=float, help='beta')
    parser.add_argument('--temperature', default=None, type=float, help='temperature')
    parser.add_argument('--channel_scaler', default=None, type=float, help='channel scaler')
    parser.add_argument('--n_conv_layers', default=None, type=int, help='n conv layers')
    parser.add_argument('--truncation_level', default=3, type=int, help='truncation level when model is fc')

    parser.add_argument('--file', type=str, help="If file is given, it's used for json config parsing.")
    parser.add_argument('--folder', type=str, help="If folder is given, it's used for json config parsing.")
    parser.add_argument('--n_runs', type=int, default=1, help="#runs")
    parser.add_argument('--least_acc', type=float, default=0.1, help="Least accuracy for hs")

    parser.add_argument('--restore_best_weight', type=bool, default=False, help="Restore best model weight")
    parser.add_argument('--use_validation_dataset', type=bool, default=False, help="Use validation dataset")

    parser.add_argument('--add_epochwise_metrics', action="store_true", default=False, help='Add epoch-wise metrics')
    parser.add_argument('--experiment_name', type=str, default=None, help='experiment_name')

    args = parser.parse_args()
    return args


def run_experiment(experiment_name, exp_dict, model_args, optimizer_args):
    model_args["device"] = torch.device("cuda")

    cl_metrics, summary_writer = train(exp_dict, model_args, optimizer_args, is_hp_search=False,
                                       least_acc=args.least_acc, add_epochwise_metrics=args.add_epochwise_metrics)

    cl_runs_folder = "runs"
    cl_metrics.save(cl_runs_folder, experiment_name, independent=False)
    summary_writer.save(cl_runs_folder, experiment_name)

    return cl_metrics


if __name__ == '__main__':
    start = time.time()
    print("Stores model and summary writer")

    args = parse_args()

    if args.file and args.folder:
        json_files = [os.path.join(args.folder, args.file)]
    elif args.file:
        json_files = [os.path.join(HYPER_PARAMS_FOLDER, args.file)]
    elif args.folder:
        json_files = [os.path.join(args.folder, k) for k in os.listdir(args.folder)]

    print("\n".join(json_files))

    for json_file_path in json_files:
        with open(json_file_path, "r") as f:
            param_dict = json.load(f)

        exp_dict = param_dict["exp_dict"].copy()
        model_args = param_dict["model_args"].copy()
        optimizer_args = param_dict["optimizer_args"].copy()

        update_config(args, exp_dict, model_args, optimizer_args)

        if args.experiment_name is None:
            experiment_name = exp_dict["dataset"]
            if args.architecture is not None:
                experiment_name += f"-network_arch_{model_args['network_arch']}"
            if args.n_conv_layers is not None:
                experiment_name += f"-n_conv_layers_{model_args['n_conv_layers']}"
            if args.n_epochs is not None:
                experiment_name += f"-n_epochs_{args.n_epochs}"
            experiment_name += f"-seed_{args.seed}"
        else:
            experiment_name = args.experiment_name

        exp_dict["experiment"] = experiment_name
        exp_dict["experiment_name"] = exp_dict["experiment"]
        exp_dict["save_task_solvers"] = True

        print(experiment_name, "\n\n")

        run_experiment(experiment_name, exp_dict, model_args, optimizer_args)
        print(json_file_path)

    stop = time.time()
    print("Total time:", stop - start)

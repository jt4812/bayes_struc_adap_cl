import argparse
from pprint import pprint

import torch

from .cl_train import train
from .eval_ll import eval_ll


def parse_args():
    parser = argparse.ArgumentParser(description='Train', add_help=True)
    parser.add_argument('--train', action="store_true", help='Whether to train model')
    parser.add_argument('--eval_ll', action="store_true", help='Whether to eval loglikelihood for model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for model training.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    exp_dict = {
        "dataset": "not_mnist",
        "n_tasks": 10,
        "batch_size": 50,
        "cl_type": "task_il",
        "seed": 9119,
        "n_epochs_per_task": 400,
        "experiment": "vae_adaptive_vcl",
        "save_task_solvers": True,
    }
    model_args = {
        "input_feature_dim": 784,
        "latent_feature_dim": 50,

        "max_width": 500,
        "n_head_layers": 2,
        "n_shared_dec_layers": 5,  # truncation_level
        "mask_type": "weight_mask",

        "a_prior": 1,
        "b_prior": 6,
        "prior_temperature": 0.5,
        "posterior_temperature": 0.5,
        "use_sigma_k": False,

        "n_train_samples": 10,
        "n_test_samples": 100,

        "act_thresh": 0.001,
        "kl_weight": 1,
        "device": torch.device("cuda"),
    }
    optimizer_args = {
        "lr_weight": args.lr,
        "lr_struc": 0.001,
        "optimizer_type": "adam",
        "decay_rate_weight": 0.9,
        "decay_rate_struc": 0.9,
        "use_single_optimizer_across_tasks": False
    }

    pprint(exp_dict)
    pprint(model_args)
    pprint(optimizer_args)

    if args.train:
        train(exp_dict, model_args, optimizer_args, is_hp_search=False)
    if args.eval_ll:
        eval_ll(exp_dict, model_args)

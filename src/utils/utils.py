def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_n_classes_tasks(dataset):
    print("Dataset: ", dataset)
    n_tasks, n_classes_per_task = None, None
    if "split_cifar10" == dataset:
        n_tasks = 5
        n_classes_per_task = 10 // n_tasks

    elif "split_cifar100" in dataset:
        n_tasks = int(dataset.split("_")[-1])
        n_classes_per_task = 100 // n_tasks

    elif "imagenet" in dataset:
        n_tasks = int(dataset.split("_")[-1])
        n_classes_per_task = 200 // n_tasks

    return n_tasks, n_classes_per_task


def update_config(args, exp_dict, model_args, optimizer_args):
    if args.seed:
        exp_dict["seed"] = args.seed

    if args.n_tasks is not None:
        exp_dict["n_tasks"] = args.n_tasks

    if args.dataset is not None:
        exp_dict["dataset"] = args.dataset

        if "cifar" in args.dataset or "imagenet" in args.dataset:
            n_tasks, n_classes = extract_n_classes_tasks(args.dataset)
            exp_dict["n_tasks"] = n_tasks
            model_args["out_feature_dim"] = n_classes

    if args.n_epochs is not None:
        exp_dict["n_epochs_per_task"] = args.n_epochs
    exp_dict["restore_best_weight"] = args.restore_best_weight
    exp_dict["use_validation_dataset"] = args.use_validation_dataset

    if args.architecture is not None:
        model_args["network_arch"] = args.architecture
    if args.channel_scaler is not None:
        model_args["channel_scaler"] = args.channel_scaler
    if args.n_conv_layers is not None:
        model_args["n_conv_layers"] = args.n_conv_layers
    if args.alpha is not None:
        model_args["a_prior"] = args.alpha
    if args.beta is not None:
        model_args["b_prior"] = args.beta
    if args.temperature is not None:
        model_args["prior_temperature"] = args.temperature
        model_args["posterior_temperature"] = args.temperature

    optimizer_args["lr_weight"] = optimizer_args["lr_weight"] if (args.lr_weight is None) else args.lr_weight
    optimizer_args["lr_struc"] = optimizer_args["lr_struc"] if (args.lr_struc is None) else args.lr_struc
    optimizer_args["decay_rate_weight"] = optimizer_args["decay_rate_weight"] if (
            args.decay_rate_weight is None) else args.decay_rate_weight

    print(exp_dict)
    print(model_args)
    print(optimizer_args)

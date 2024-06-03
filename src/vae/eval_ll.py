import os

import numpy as np
import torch.nn as nn

from src.data.cl_datasets import get_cl_dataloaders
from src.utils.save_model_utils import get_trained_solver
from src.utils.cl_metric_accumulator import CLResults
from src.utils.visualize import plot_images, save_images
from .eval_utils import calculate_lower_bound
from .models.vae_adaptive import BayesAdaptiveVAE
from .train_eval_utils import generate_samples_until_now


def _eval_ll(exp_dict, model, get_dataloader, loss_fn):
    test_dataloaders = []
    metrics = CLResults()

    for t_idx in range(exp_dict["n_tasks"]):
        print("Task: ", t_idx)
        model = get_trained_solver(t_idx, model.model_args, exp_dict, exp_name=exp_dict["experiment"])
        *_, cur_test_dataloader = get_dataloader(t_idx)
        test_dataloaders.append(cur_test_dataloader)

        tasks_test_metrics = [
            calculate_lower_bound(model, test_dataloader, t_idx_, loss_fn, n_samples=5000)
            for t_idx_, test_dataloader in enumerate(test_dataloaders)]
        tasks_test_elbo = [t[0] for t in tasks_test_metrics]
        # tasks_test_lls = [t[1] for t in tasks_test_metrics]
        tasks_test_kls = [t[1] for t in tasks_test_metrics]

        metrics.add_result(tasks_test_elbo)
        print("Test data likelihoods:", tasks_test_elbo)
        print("Test kl divergences:", tasks_test_kls)

        save_path = os.path.join(exp_dict["experiment"], "mean_metric_acc_task.png")
        metrics.plot_save_metrics_across_tasks(save_path)
        print(metrics)

        gen_samples_ls = generate_samples_until_now(model, t_idx, n_samples=100)
        save_images(exp_dict["experiment"], exp_dict["dataset"], gen_samples_ls)

        x_list = [gen_samples_ls[i][:1] for i in range(len(gen_samples_ls))]
        x_list = np.concatenate(x_list, 0)
        tmp = np.zeros([10, 784])
        tmp[:t_idx + 1] = x_list
        if t_idx == 0:
            x_gen_all = tmp
        else:
            x_gen_all = np.concatenate([x_gen_all, tmp], 0)
        plot_images(x_gen_all, (28, 28), exp_dict["experiment"], dataset=exp_dict["dataset"])
    return metrics


def eval_ll(exp_dict, model_args):
    get_dataloader, _ = get_cl_dataloaders(exp_dict, exp_dict["batch_size"])
    model = BayesAdaptiveVAE(model_args, exp_dict).to(model_args["device"])
    loss_fn = nn.BCELoss(reduction="none")
    cl_metrics = _eval_ll(exp_dict, model, get_dataloader, loss_fn, )
    return cl_metrics, model

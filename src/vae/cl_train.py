import os

import numpy as np
import torch.nn as nn

from src.data.cl_datasets import get_cl_dataloaders
from src.utils.cl_metric_accumulator import CLResults
from src.utils.optimizer import get_optimizer
from src.utils.save_model_utils import save_solver_model
from src.utils.utils import seed_everything
from src.utils.visualize import plot_images, save_images
from .eval_utils import calculate_lower_bound
from .models.vae_adaptive import BayesAdaptiveVAE
from .train_eval_utils import generate_samples_until_now, train_single_epoch_bayes_adaptive


def cl_train(exp_dict, model, get_dataloader, loss_fn, optimizer_args, is_hp_search=True, interval_print=10):
    test_dataloaders = []
    metrics = CLResults()

    for t_idx in range(exp_dict["n_tasks"]):
        print("Task: ", t_idx)
        optimizer = get_optimizer(model, optimizer_args, t_idx)
        cur_train_dataloader, _, cur_test_dataloader = get_dataloader(t_idx)
        test_dataloaders.append(cur_test_dataloader)

        for epoch in range(exp_dict["n_epochs_per_task"]):
            train_loss, neg_loglike, latent_kld, weight_kld, kld_struc = train_single_epoch_bayes_adaptive(
                model, cur_train_dataloader, optimizer, loss_fn, task_idx=t_idx, is_hp_search=is_hp_search)
            if (not is_hp_search) and (epoch % interval_print == 0):
                print(
                    f"{epoch + 1}/{exp_dict['n_epochs_per_task']}",
                    "train_loss:", train_loss,
                    "neg_loglike:", neg_loglike,
                    "latent_kld:", latent_kld,
                    "weight_kld:", weight_kld,
                    "kld_struc:", kld_struc,
                )

        if not is_hp_search:
            # Lower Bounds
            tasks_test_lower_bounds = [
                calculate_lower_bound(model, test_dataloader, t_idx_, loss_fn, n_samples=5000)
                for t_idx_, test_dataloader in enumerate(test_dataloaders)]
            tasks_test_lbs = [t[0] for t in tasks_test_lower_bounds]
            # tasks_test_kl_latents = [t[1] for t in tasks_test_lower_bounds]

            metrics.add_result(tasks_test_lbs)
            print(metrics)

            save_path = os.path.join(exp_dict["experiment"], "mean_metric_acc_task.png")
            metrics.plot_save_metrics_across_tasks(save_path)

            # Image Generation
            gen_samples_ls = generate_samples_until_now(model, t_idx, n_samples=100)
            save_images(exp_dict["experiment"], exp_dict["dataset"], gen_samples_ls)

            # Stack of Images Generation across Training
            x_list = [gen_samples_ls[i][:1] for i in range(len(gen_samples_ls))]
            x_list = np.concatenate(x_list, 0)
            tmp = np.zeros([10, 784])
            tmp[:t_idx + 1] = x_list
            if t_idx == 0:
                x_gen_all = tmp
            else:
                x_gen_all = np.concatenate([x_gen_all, tmp], 0)
            plot_images(x_gen_all, (28, 28), exp_dict["experiment"], dataset=exp_dict["dataset"])

        if exp_dict["save_task_solvers"]:
            save_solver_model(t_idx, model, exp_dict["experiment"])

        model.reset_for_new_task()

    if is_hp_search:
        tasks_test_lower_bounds = [
            calculate_lower_bound(model, test_dataloader, t_idx_, loss_fn, n_samples=5000)
            for t_idx_, test_dataloader in enumerate(test_dataloaders)]
        tasks_test_lbs = [t[0] for t in tasks_test_lower_bounds]

        metrics.add_result(tasks_test_lbs)
        print(metrics)
    return metrics


def train(exp_dict, model_args, optimizer_args, is_hp_search=True):
    seed_everything(exp_dict["seed"])

    get_dataloader = get_cl_dataloaders(exp_dict, exp_dict["batch_size"])
    if len(get_dataloader) > 1:
        get_dataloader = get_dataloader[0]

    model = BayesAdaptiveVAE(model_args, exp_dict).to(model_args["device"])

    if optimizer_args["use_single_optimizer_across_tasks"]:
        optimizer_args = get_optimizer(model, **optimizer_args)

    loss_fn = nn.BCELoss(reduction="none")

    cl_metrics = cl_train(exp_dict, model, get_dataloader, loss_fn, optimizer_args, is_hp_search=is_hp_search)

    metrics = {
        "final_max_cl_metric": cl_metrics.final_max_min_metrics[0],
        "final_min_cl_metric": cl_metrics.final_max_min_metrics[1],
        "final_mean_metric": cl_metrics.final_mean_metric
    }

    return metrics, cl_metrics, model

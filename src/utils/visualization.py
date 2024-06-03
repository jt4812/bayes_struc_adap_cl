import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_individual_accuracy_together(task_wise_epoch_test_accs):
    x_idx = np.arange(1, len(task_wise_epoch_test_accs[0]) + 1)

    plt.figure(figsize=(6, 5))

    for accs in task_wise_epoch_test_accs:
        plt.plot(x_idx[-len(accs):], accs)

    plt.grid()
    plt.xlabel("Training time")
    plt.ylabel("Accuracy")
    plt.show()


def plot_individual_accuracy(task_wise_epoch_test_accs, save_folder=None):
    x_idx = np.arange(1, len(task_wise_epoch_test_accs[0]) + 1)
    n_tasks = len(task_wise_epoch_test_accs)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FF5733', '#33FF57', '#337DFF', '#FF33C7', '#33FFD6', '#FFD633',
              '#8B4513', '#9932CC', '#20B2AA', '#FF6347', '#008080', '#DA70D6']

    fig, axs = plt.subplots(n_tasks, 1, figsize=(6, 5))

    for idx, accs in enumerate(task_wise_epoch_test_accs):
        axs[idx].plot(x_idx[-len(accs):], accs, color=colors[idx])
        axs[idx].set_xticks(x_idx)
        axs[idx].set_ylabel(f"Task {idx}")
        #     axs[idx].axis('off')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['bottom'].set_visible(False)
        axs[idx].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
    plt.xlabel("Training Time")
    fig.suptitle("Accuracy")
    fig.tight_layout()
    if save_folder is not None:
        plt.savefig(f"{save_folder}/individual_accuracy.png")
    plt.show()


def plot_ak_bk(ak_bk_ls, layer_idx, save_folder=None):
    plt.figure(figsize=(10, 6))
    ak_bk_ls_layer = ak_bk_ls[layer_idx]
    ak = ak_bk_ls_layer[:, :, 0].ravel()
    bk = ak_bk_ls_layer[:, :, 1].ravel()
    plt.plot(ak)
    plt.plot(bk)

    gap = ak_bk_ls_layer[:, :, 0].shape[-1]
    for k in np.arange(0, len(ak), gap):
        plt.axvline(x=k, color='g', linestyle='--', linewidth=0.6)

    plt.title(f"Layer {layer_idx}")
    plt.legend(["ak", "bk"])
    plt.xlabel("Training time")
    if save_folder is not None:
        plt.savefig(f"{save_folder}/ak_bk_layer_{layer_idx}.png")
    plt.show()


def plot_beta_pi(beta_pi_ls, layer_idx, save_folder=None):
    plt.figure(figsize=(10, 6))
    beta_pi = beta_pi_ls[layer_idx]
    plt.plot(beta_pi.ravel())

    gap = beta_pi.shape[-1]
    for k in np.arange(0, len(beta_pi.ravel()), gap):
        plt.axvline(x=k, color='g', linestyle='--', linewidth=0.6)

    plt.title(f"Layer {layer_idx}")
    plt.xlabel("Training time")
    if save_folder is not None:
        plt.savefig(f"{save_folder}/beta_pi_layer_{layer_idx}.png")
    plt.show()


def plot_model_usage(model_usage_ls, layer_idx, save_folder=None):
    plt.figure(figsize=(10, 6))
    model_usage = model_usage_ls[layer_idx]
    plt.plot(model_usage.ravel())

    gap = model_usage.shape[-1]
    for k in np.arange(0, len(model_usage.ravel()), gap):
        plt.axvline(x=k, color='g', linestyle='--', linewidth=0.6)

    plt.title(f"Layer {layer_idx}")
    plt.xlabel("Training time")
    if save_folder is not None:
        plt.savefig(f"{save_folder}/model_usage_layer_{layer_idx}.png")
    plt.show()


def plot_model_usage_used(model_usage_used_ls, layer_idx, save_folder=None):
    model_usage = model_usage_used_ls[layer_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(model_usage)
    plt.title(f"Layer {layer_idx}")
    plt.xlabel("Training time")
    plt.xticks(range(len(model_usage_used_ls)))
    plt.grid()
    if save_folder is not None:
        plt.savefig(f"{save_folder}/model_usage_used_layer_{layer_idx}.png")
    plt.show()


def plot_heatmap(iou_inter_task, layer_idx, save_folder=None):
    n_tasks = len(iou_inter_task[0])
    f, ax = plt.subplots(figsize=(0.6 * n_tasks, 0.6 * n_tasks))

    sns.heatmap(
        iou_inter_task[1:, :-1],
        annot=True,
        fmt=".0f",
        linewidths=.5,
        ax=ax,
        cbar=False,
        cmap="Greens",
        vmin=0,
        vmax=100,
        linecolor="#eaeaea"
    )
    ax.set_xticklabels(np.arange(0, n_tasks - 1))
    ax.set_yticklabels(np.arange(1, n_tasks))

    ax.set_xlabel("Task")
    ax.set_ylabel("Task")

    plt.title(f"Layer {layer_idx}")
    if save_folder is not None:
        plt.savefig(f"{save_folder}/iou_inter_task_layer_{layer_idx}.png")
    plt.show()


def generate_plots(cl_metrics, summary_writer, save_folder=None):
    try:
        epochwise_metrics = cl_metrics.epochwise_metrics
        n_tasks = len(epochwise_metrics[-1])

        # Taskwise Accuracy
        task_wise_epoch_test_accs = [[] for _ in range(n_tasks)]

        for accs in cl_metrics.epochwise_metrics:
            for task_idx, acc in enumerate(accs):
                task_wise_epoch_test_accs[task_idx].append(acc)
        plot_individual_accuracy(task_wise_epoch_test_accs, save_folder)

        # Model Params Evolution
        ak_bk_ls = np.array(summary_writer.ak_bk_ls)
        for layer_idx in range(len(ak_bk_ls)):
            plot_ak_bk(ak_bk_ls, layer_idx, save_folder)

        # Per Layer Activation from Beta Process
        beta_pi_ls = np.array(summary_writer.beta_pi_ls)
        for layer_idx in range(len(beta_pi_ls)):
            plot_beta_pi(beta_pi_ls, layer_idx, save_folder)

        # Per Layer Activation
        model_usage_ls = np.array(summary_writer.model_usage_ls)

        for layer_idx in range(len(model_usage_ls)):
            plot_model_usage(model_usage_ls, layer_idx, save_folder)

        # Model Usage
        model_usage_used_ls = summary_writer.model_usage_used_ls
        for layer_idx in range(len(model_usage_used_ls)):
            plot_model_usage_used(model_usage_used_ls, layer_idx, save_folder)

        # IoU between task specific masks
        iou_inter_task_ls = summary_writer.iou_inter_task
        for layer_idx in range(len(iou_inter_task_ls)):
            iou_inter_task_ = iou_inter_task_ls[layer_idx]
            plot_heatmap(iou_inter_task_, layer_idx, save_folder)
    except:
        print("Some plots could not be generated")

import math
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def visualize_pis(pi_ls, threshold_ls, exp_name):
    threshold = max(threshold_ls)
    n_tasks = len(pi_ls)

    font = {'size': 18}
    matplotlib.rc('font', **font)

    fig_save_path = os.path.join("reports", exp_name + ".png")
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)

    plt.figure(figsize=(14, 6))
    x_range = np.arange(1, n_tasks + 1)

    pi_ls = np.array(pi_ls)

    # pi_ls = pi_ls[:, :, 0, 0, :].mean(1)

    for pis_ in pi_ls.T[:threshold]:
        plt.plot(x_range, pis_, "s-.", markersize=10)

    plt.legend([f"Layer {i + 1}" for i in range(len(pi_ls.T))])
    plt.xticks(x_range)
    plt.xlabel("Task")
    plt.ylabel("Activation Probability")
    plt.grid(alpha=0.25)

    plt.savefig(fig_save_path)

    plt.show()


def get_save_path(exp_name, dataset, task_idx, task_idx_=None):
    save_dir = os.path.join(exp_name, dataset)
    if task_idx_ is not None:
        filename = str(task_idx) + "-" + str(task_idx_) + ".png"
    else:
        filename = str(task_idx) + ".png"
    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    return save_path


def save_images(exp_name, dataset, gen_samples_ls):
    for t_idx, gen_samples in enumerate(gen_samples_ls):
        gen_samples = gen_samples.reshape(-1, 1, 28, 28)
        grid_img = make_grid(gen_samples, nrow=10, padding=10, pad_value=0.5)
        plt.imshow(grid_img.permute(1, 2, 0))
        save_path = get_save_path(exp_name, dataset, len(gen_samples_ls) - 1, t_idx)
        plt.axis('off')
        plt.savefig(save_path)


def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'

    def cell(i, j):
        ind = i * n_cols + j
        if i * n_cols + j < array.shape[0]:
            return array[ind].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def plot_images(images, shape, exp_name, dataset, n_rows=10, color=True):
    images = reshape_and_tile_images(images, shape, n_rows)
    save_path = os.path.join(exp_name, dataset + "_gen_all.png")
    if color:
        from matplotlib import cm
        plt.imsave(fname=save_path, arr=images, cmap=cm.Greys_r)
    else:
        plt.imsave(fname=save_path, arr=images, cmap='Greys')
    print("saving image to " + save_path)
    plt.close()


def visualize_masks(model, exp_dict, taskwise_threshold_pi_ls, network_usages, t_idx, fig1, ax1, fig2, ax2):
    if model.use_fixed_mask:
        Z_mask = model.task_mask_first_layer_list[t_idx], model.task_mask_remaining_layer_list[t_idx]
        ax1[t_idx].imshow(Z_mask[0].mT.detach().cpu().numpy(), vmin=0.3, vmax=0.75, cmap="Blues")
        ax1[t_idx].set_xticks([])
        ax1[t_idx].set_yticks([])
        ax1[t_idx].set_xlabel(f"Task {t_idx + 1}", fontsize=18)
        fig1.suptitle("Task Specific Masks", fontsize=22)
        fig1.savefig(os.path.join("reports", exp_dict['experiment'] + "_task_mask.png"))

        usage = [(Z_mask[0].sum() / Z_mask[0].numel()).item()]
        if Z_mask[1].numel() > 0:
            usage_remaining_layers = (Z_mask[1].sum(dim=1).sum(dim=1) / Z_mask[1][0].numel()).cpu().numpy()
            usage += usage_remaining_layers.tolist()

        network_usages.append(usage)
        taskwise_threshold_pi_ls.append((-1, np.array(network_usages)))

        print("Current task weights usage:", usage)

        model.reset_for_new_task()

        Z_mask_union = model.union_mask
        ax2[t_idx].imshow(Z_mask_union[0].mT.detach().cpu().numpy(), vmin=0, vmax=1, cmap="Blues")
        fig2.savefig(os.path.join("reports", exp_dict['experiment'] + "_union_mask.png"))
        print("Cumulative task weights usage:",
              (Z_mask_union[0].sum() / Z_mask_union[0].numel()).item(),
              (Z_mask_union[1].sum(dim=1).sum(dim=1) / Z_mask_union[1][0].numel()).cpu().numpy() \
                  if Z_mask_union[1].numel() > 0 else ''
              )
        print()
        print()
    else:
        model.eval()
        Z_mask, threshold, pi = model.generate_single_mask()
        taskwise_threshold_pi_ls.append((threshold, pi.detach().cpu().numpy()))
        print("pis: ", np.round(pi.detach().cpu().numpy()[:7] * 100, 2))
        model.train()

        ax1[t_idx].imshow(Z_mask[0].T.detach().cpu().numpy(), vmin=0, vmax=1)
        fig1.savefig(os.path.join("reports", "task_mask.png"))
        print("Current task weights usage:",
              (Z_mask[0].sum() / Z_mask[0].numel()).item(),
              (Z_mask[1][0].sum(dim=1).sum(dim=1) / Z_mask[1][0][0].numel()).cpu().numpy() \
                  if Z_mask[1].numel() > 0 else ''
              )

        model.reset_for_new_task()

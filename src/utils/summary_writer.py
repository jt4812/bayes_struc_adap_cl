import os.path
import pickle

import numpy as np


class SummaryWriter:
    """ Creates iteration wise/ epochwise/taskwise summary of the training process.
    ================================================================
    Logs the following model parameters:
    1. alpha, beta - iteration, epoch and task wise - per layer
    2. beta pi (from beta distribution) - iteration, epoch and task wise - per layer
    3. bernoulli pi (includes sigma effect / actual activation) - iteration, epoch and task wise - per layer
    6. no of active layers (training) - iteration, epoch and task wise

    4. model usage change - task wise - per layer
    7. no of active layers (task specific mask) - task wise
    3. bernoulli pi (task specific mask) - iteration, epoch and task wise - per layer

    8. overlap between task specific masks - end

    ================================================================
    Logs the following metrics:
    1. Training Accuracy - iteration, epoch and task wise
    2. Average Test Accuracy - task wise
    3. Test Accuracy - epoch wise
    4. Backward Transfer - task wise

    learning rate - epoch and task wise
    """

    def __init__(self, n_layers, n_tasks):
        self.tensorboard_writer = None
        self.task_idx = 0
        self.epoch_idx = 0

        self.n_layers = n_layers

        self.ak_bk_ls = [[[] for _ in range(n_tasks)] for _ in range(n_layers)]
        self.beta_pi_ls = [[[] for _ in range(n_tasks)] for _ in range(n_layers)]

        self.beta_pi_ls_task_end = [[] for _ in range(n_tasks)]

        self.model_usage_ls = [[[] for _ in range(n_tasks)] for _ in range(n_layers)]
        self.active_layers_training = [[] for _ in range(n_tasks)]

        self.active_layers_inferred = [0 for _ in range(n_tasks)]

        self.tsm_activation_ls = [[0 for _ in range(n_tasks)] for _ in range(n_layers)]
        self.model_usage_used_ls = [[0 for _ in range(n_tasks)] for _ in range(n_layers)]
        self.learning_rates = [[] for _ in range(n_tasks)]

        self.iou_inter_task = []

    def log_model_params(self, task_idx, epoch_idx, model):
        self.task_idx, self.epoch_idx = task_idx, epoch_idx
        self.log_training_structure_params(model)

    def log_training_structure_params(self, model, **kwargs):
        """ Logs the variational parameters of the model during training """
        # 1. alpha, beta - iteration, epoch and task wise - per layer
        # 2. beta pi (from beta distribution) - iteration, epoch and task wise - per layer
        # 3. conbernoulli pi (includes sigma effect / actual activation) - iteration, epoch and task wise - per layer

        training_status = model.training
        model.eval()
        Z_single, threshold, beta_pis, model_usage = model.generate_single_mask()
        if training_status:
            model.train()

        ak, bk = model.structure_sampler.variational_params
        ak = ak.detach().cpu().numpy()  # K
        bk = bk.detach().cpu().numpy()  # K

        beta_pis = beta_pis.detach().cpu().numpy()  # K

        # 6. no of active layers (training) - iteration, epoch and task wise

        for layer_idx in range(model.n_layers):
            self.ak_bk_ls[layer_idx][self.task_idx].append([ak[layer_idx], bk[layer_idx]])
            self.beta_pi_ls[layer_idx][self.task_idx].append(beta_pis[layer_idx])

            self.model_usage_ls[layer_idx][self.task_idx].append(model_usage[layer_idx])

        self.active_layers_training[self.task_idx].append(threshold)

    def log_test_structure_params(self, model, is_final_task):
        """ Logs the variational parameters of the model after the model completed training a task """

        if not(hasattr(model, "task_mask_list") or hasattr(model, "task_mask_first_layer_list")):
            print("Don't log test structure params for models with task masks except for beta pi")
            training_status = model.training
            model.eval()
            Z_single, threshold, beta_pis, model_usage = model.generate_single_mask()
            if training_status:
                model.train()

            beta_pis = beta_pis.detach().cpu().numpy()  # K
            self.beta_pi_ls_task_end[self.task_idx] = beta_pis
            return
        # 4. model usage change - task wise - per layer

        # self.task_mask_first_layer_list[task_idx].data.copy_(Z_single[0].squeeze().detach().data)
        #                 self.task_mask_remaining_layer_list[task_idx].data.copy_(Z_single[1].squeeze().detach().data)
        if hasattr(model, "task_mask_list"):
            task_mask_list = model.task_mask_list
        elif hasattr(model, "task_mask_first_layer_list"):
            task_mask_list = [[] for _ in range(model.n_tasks)]
            for task in range(model.n_tasks):
                task_mask_list[task].append(model.task_mask_first_layer_list[task])
                for mask_remaining in model.task_mask_remaining_layer_list[task]:
                    task_mask_list[task].append(mask_remaining)


        task_mask_ls = task_mask_list[self.task_idx]
        activation_prob = [mask_.sum() / mask_.numel() for mask_ in task_mask_ls]

        union_mask = model.union_mask
        if (len(union_mask) == 2) and (len(union_mask[0].shape) != len(union_mask[1].shape)):
            union_mask = [union_mask[0]] + [k for k in union_mask[1]]

        model_usage = [(mask_.sum() / mask_.numel()).cpu().item() for mask_ in union_mask]

        for layer_idx in range(model.n_layers):
            self.tsm_activation_ls[layer_idx][self.task_idx] = activation_prob[layer_idx]
            self.model_usage_used_ls[layer_idx][self.task_idx] = model_usage[layer_idx]

        # 7. no of active layers (task specific mask) - task wise

        activated_layers = model.task_threshold_list[self.task_idx].item()
        self.active_layers_inferred[self.task_idx] = activated_layers

        # 8. overlap between task specific masks - end
        if is_final_task:
            for layer_idx in range(model.n_layers):
                try:
                    self.iou_inter_task.append(self.get_iou_inter_task(layer_idx, task_mask_list, model.n_tasks))
                except:
                    pass

    def log_learning_rate(self, optimizer):
        """ Logs the learning rate of the optimizer during training """
        self.learning_rates[self.task_idx].append(optimizer.param_groups[0]['lr'])

    @staticmethod
    def get_iou_inter_task(layer_idx, task_mask_list, n_tasks):
        iou_inter_task = np.full((n_tasks, n_tasks), np.nan)

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i <= j:
                    iou_inter_task[i][j] = None
                    break

                overlap_count = (task_mask_list[i][layer_idx] * task_mask_list[j][layer_idx]).sum()
                union_count = ((task_mask_list[i][layer_idx] + task_mask_list[j][layer_idx]) > 0).sum()

                iou = overlap_count / (union_count + 1e-7)

                iou_inter_task[i][j] = round(iou.item() * 100)
        return iou_inter_task

    def save(self, save_dir, experiment_name):
        save_path = os.path.join(save_dir, experiment_name, "params_summary.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

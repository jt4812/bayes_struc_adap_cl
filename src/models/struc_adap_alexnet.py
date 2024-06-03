import torch
from torch import nn

from src.layers.bayes_conv import BayesConvBlock
from src.layers.bayes_layer import BayesLinear, BayesMLPBlock
from src.models.structure_sampler_conv import NetworkStructureSampler


class BayesAdaptiveCNN(nn.Module):
    def __init__(self, args, exp_dict=None):
        super(BayesAdaptiveCNN, self).__init__()
        self.model_args = args
        self.exp_dict = exp_dict

        self.out_feature_dim = args["out_feature_dim"]

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.posterior_std = args["posterior_std"]

        self.device = args["device"]

        self.kl_weight = args["kl_weight"]
        self.has_norm = args["has_norm"]

        self.has_bayesian_head = args["has_bayesian_head"]
        self.use_fixed_mask = args["use_fixed_mask"]

        self.single_head = exp_dict["single_head"]
        self.n_tasks = exp_dict["n_tasks"]

        self.mask_type = args["mask_type"]  # ["neuron_mask", "weight mask"]

        self.network_arch = args["network_arch"]
        self.use_kl_mask = args["use_kl_mask"]
        self.channel_scaler = args["channel_scaler"] if "channel_scaler" in args else 1
        assert self.channel_scaler in [0.25, 0.5, 1]

        self.conv_mask_type = args["conv_mask_type"]
        self.reestimate_beta_posterior = False

        self.build_model()

    def build_model(self):
        self.deepest_activated_layer = -1
        if "use_task_specific_norms" in self.model_args:
            self.use_task_specific_norms = self.model_args["use_task_specific_norms"]

            if "transfer_bn_params" in self.model_args:
                self.transfer_bn_params = self.model_args["transfer_bn_params"]

        self.conv_layers = nn.ModuleList(
            [
                BayesConvBlock(3, int(128 * self.channel_scaler), (4, 4), (2, 2), "valid", self.posterior_std,
                               has_norm=self.has_norm, residual=False, has_max_pool=True,
                               use_task_specific_norms=self.use_task_specific_norms, n_tasks=self.n_tasks),
                BayesConvBlock(int(128 * self.channel_scaler), int(256 * self.channel_scaler), (3, 3), (2, 2), "valid",
                               self.posterior_std,
                               has_norm=self.has_norm, residual=False, has_max_pool=True,
                               use_task_specific_norms=self.use_task_specific_norms, n_tasks=self.n_tasks),
                BayesConvBlock(int(256 * self.channel_scaler), int(512 * self.channel_scaler), (2, 2), (2, 2), "valid",
                               self.posterior_std,
                               has_norm=self.has_norm, residual=False, has_max_pool=True,
                               use_task_specific_norms=self.use_task_specific_norms, n_tasks=self.n_tasks)
            ])
        fc_width = int(2048 * self.channel_scaler)
        self.fc_layers = nn.ModuleList(
            [
                BayesMLPBlock(fc_width, fc_width, posterior_std=self.posterior_std, residual=True),
                BayesMLPBlock(fc_width, fc_width, posterior_std=self.posterior_std, residual=True)
            ])

        self.n_heads = 1 if self.single_head else self.n_tasks

        if self.has_bayesian_head:
            self.out_layer = nn.ModuleList([
                BayesLinear(fc_width, self.out_feature_dim, posterior_std=self.posterior_std) for _ in
                range(self.n_heads)
            ])
        else:
            self.out_layer = nn.ModuleList([
                nn.Linear(fc_width, self.out_feature_dim) for _ in range(self.n_heads)
            ])

        conv_weight_shape_ls = []
        for layer_ in self.conv_layers:
            conv_weight_shape_ls.append(layer_.get_weight_shape())

        fc_weight_shape_ls = []
        for layer_ in self.fc_layers:
            fc_weight_shape_ls.append(layer_.get_weight_shape())

        weight_shape_ls = [shape[:2] if self.conv_mask_type == "single_per_channel" else shape for shape in
                           conv_weight_shape_ls + fc_weight_shape_ls]
        if self.use_fixed_mask:
            self.task_mask_list = nn.ParameterList([
                nn.ParameterList([nn.Parameter(torch.ones(shape), requires_grad=False) for shape in weight_shape_ls])
                for _ in range(self.n_tasks)
            ])

            self.task_threshold_list = nn.Parameter(torch.ones(self.n_tasks, 1, dtype=torch.int) * -1,
                                                    requires_grad=False)

            if self.use_kl_mask:
                self.union_mask = None

        self.n_layers = len(self.conv_layers) + len(self.fc_layers)

        # define neural network structure sampler with parameters defined in argument parser
        self.infer_model_structure = self.model_args["infer_model_structure"]
        if self.infer_model_structure:
            self.structure_sampler = NetworkStructureSampler(self.model_args,
                                                             (conv_weight_shape_ls, fc_weight_shape_ls))

    def _forward(self, x, mask_matrix, threshold, head_idx=0, task_idx=None):
        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        self.deepest_activated_layer = threshold
        conv_mask = mask_matrix[:len(self.conv_layers)]
        fc_mask = mask_matrix[len(self.conv_layers):]

        for layer, mask in zip(self.conv_layers, conv_mask):
            x = layer(x, mask, task_idx)

        batch_size, C, H, W = x.shape
        x = x.view(batch_size, -1)

        for layer, mask in zip(self.fc_layers, fc_mask):
            x = layer(x, mask)

        if head_idx == -1:
            # run model to predict output from all the heads
            out = [self.out_layer[h_idx](x) for h_idx in range(self.n_heads)]
            out = torch.stack(out)
        else:
            out = self.out_layer[head_idx](x)
        return out

    def forward(self, x, task_idx=None, n_samples=None, use_fixed_mask=False, use_all_heads=False, ret_threshold=False,
                use_single_head=False):
        n_samples = (self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        if self.use_fixed_mask:
            act_vec = self.forward_masked(x, task_idx, n_samples, use_fixed_mask, use_all_heads, ret_threshold)
        else:
            act_vec = self.forward_unmasked(x, task_idx, n_samples, use_all_heads, ret_threshold, use_single_head)

        return act_vec

    def forward_unmasked(self, x, task_idx=None, n_samples=None, use_all_heads=False,
                         ret_threshold=False, use_single_head=False):
        act_vec = []

        if self.infer_model_structure:
            Z, threshold = self.structure_sampler(n_samples)

        # task_range = task_idx if use_single_head else (-1 if use_all_heads else self.get_head_idx(task_idx))
        task_range = range(self.n_tasks) if use_all_heads else [task_idx]

        for s in range(n_samples):

            if self.infer_model_structure:
                Z_mask = [Z_l[s] for Z_l in Z]
            else:
                Z_mask = [None for _ in range(len(self.conv_layers) + len(self.fc_layers))]
                threshold = len(self.conv_layers) + len(self.fc_layers)

            act_vec_sample = []

            for task_idx_ in task_range:
                out = self._forward(x, Z_mask, threshold, head_idx=task_idx_, task_idx=task_idx)
                act_vec_sample.append(out.unsqueeze(0))
            act_vec_sample = torch.cat(act_vec_sample, dim=0)
            act_vec.append(act_vec_sample.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)  # n_samples x n_tasks x batch_size x n_classes

        # squeeze first dim if only one task
        act_vec = act_vec.squeeze(dim=1)

        if ret_threshold:
            return act_vec, threshold
        else:
            return act_vec

    def forward_masked(self, x, task_idx=None, n_samples=None, use_fixed_mask=False, use_all_heads=False,
                       ret_threshold=False):
        n_samples = (self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        act_vec = []

        if not use_fixed_mask:
            Z, threshold = self.structure_sampler(n_samples)

        task_range = range(self.n_tasks) if use_all_heads else [task_idx]

        for s in range(n_samples):
            if not use_fixed_mask:
                Z_mask = [Z_l[s] for Z_l in Z]

            act_vec_sample = []

            for task_idx_ in task_range:
                if use_fixed_mask:
                    Z_mask = self.task_mask_list[task_idx_]
                    threshold = len(self.conv_layers) + len(self.fc_layers)

                out = self._forward(x, Z_mask, threshold, head_idx=task_idx_, task_idx=task_idx)
                act_vec_sample.append(out.unsqueeze(0))
            act_vec_sample = torch.cat(act_vec_sample, dim=0)
            act_vec.append(act_vec_sample.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)  # n_samples x n_tasks x batch_size x n_classes

        # squeeze first dim if only one task
        act_vec = act_vec.squeeze(dim=1)

        if ret_threshold:
            return act_vec, threshold
        else:
            return act_vec

    def get_E_loglike(self, neg_loglike_fun, y_preds, y_true):
        n_samples = self.n_train_samples

        batch_sze = len(y_true)
        y_true = y_true.squeeze().repeat(n_samples)
        y_preds = y_preds.view(n_samples * batch_sze, -1)

        neg_loglike = neg_loglike_fun(y_preds, y_true)
        E_neg_loglike = neg_loglike.mean()
        return E_neg_loglike

    def estimate_total_loss(self, neg_loglike_fun, y_preds, y_true, kl_scale=1, use_fixed_mask=False, task_idx=None,
                            kd_loss_fun=None):
        # the case that task id is present and all heads are predicted occur when
        # y_preds shape: n_samples x batch_size x n_classes
        E_neg_loglike = self.get_E_loglike(neg_loglike_fun, y_preds, y_true)

        weight_kld = self.kl_divergence_weight(task_idx)

        if self.infer_model_structure:
            struct_kld = self.structure_sampler.kl_divergence

            if use_fixed_mask:
                kld = weight_kld
            else:
                kld = weight_kld + struct_kld
        else:
            struct_kld = torch.tensor([0]).to(self.device)
            kld = weight_kld

        # KL term weighted by kl_scale for ELBO consistency in mini batch and kl_weight for regularization effectiveness
        total_loss = E_neg_loglike + kld * kl_scale * self.kl_weight
        return total_loss, E_neg_loglike, weight_kld * kl_scale * self.kl_weight, struct_kld * kl_scale * self.kl_weight

    def get_E_loglike_extended(self, neg_loglike_fun, y_preds, y_true, task_ids, non_current_class_loss_wt=None):
        n_samples = self.n_train_samples
        neg_loglike = 0

        for task_idx in torch.unique(task_ids):
            _task_indices = (task_ids == task_idx)
            y_true_cur_task = y_true[_task_indices]

            # task_idx at index 1 selects head with cur_task idx
            y_preds_cur_task = y_preds[:, task_idx, _task_indices]
            y_preds_cur_task_expanded = y_preds_cur_task.reshape(n_samples * sum(_task_indices), -1)
            y_true_cur_task = y_true_cur_task.squeeze().repeat(n_samples)

            current_task_loss = neg_loglike_fun(y_preds_cur_task_expanded, y_true_cur_task)

            if non_current_class_loss_wt is not None:
                other_task_indices = [i for i in range(self.n_tasks) if i != task_idx]
                y_preds_non_cur = y_preds[:, other_task_indices]
                y_preds_non_cur = y_preds_non_cur[:, :, _task_indices]
                other_tasks_loss = neg_loglike_fun(y_preds_non_cur, torch.ones_like(y_preds_non_cur) * 0.5)

                neg_loglike += (current_task_loss.mean() + non_current_class_loss_wt
                                * other_tasks_loss.mean()) / (1 + non_current_class_loss_wt)
            else:
                neg_loglike += current_task_loss.mean()

        E_neg_loglike = neg_loglike / (task_idx + 1)
        return E_neg_loglike

    def get_head_idx(self, task_idx):
        if self.single_head:
            head_idx = 0  # use head at idx 0 irrespective of cl type
        elif self.exp_dict["use_replay"]:
            head_idx = -1  # use all the heads to generate results
        elif self.exp_dict["cl_type"] == "task_il":
            head_idx = task_idx
        elif self.exp_dict["cl_type"] == "domain_il":
            # head_idx = task_idx
            raise NotImplemented()
        elif self.exp_dict["cl_type"] == "class_il":
            head_idx = -1
        else:
            raise NotImplemented()
        return head_idx

    def kl_divergence_weight(self, task_idx=None):
        weight_kld = torch.tensor([0]).float().to(self.device)

        if self.use_kl_mask and self.union_mask is not None:
            conv_mask = self.union_mask[:len(self.conv_layers)]
            fc_mask = self.union_mask[len(self.conv_layers):]

            for layer, mask in zip(self.conv_layers, conv_mask):
                weight_kld += layer.masked_kl_divergence(mask)

            for layer, mask in zip(self.fc_layers, fc_mask):
                weight_kld += layer.masked_kl_divergence(mask)

        else:
            for layer in self.conv_layers:
                weight_kld += layer.kl_divergence
            for layer in self.fc_layers:
                weight_kld += layer.kl_divergence

        if self.has_bayesian_head:
            if self.single_head:
                weight_kld += self.out_layer[0].kl_divergence
            else:
                weight_kld += self.out_layer[task_idx].kl_divergence

        return weight_kld

    def kl_divergence(self, task_idx=None):
        weight_kld = self.kl_divergence_weight(task_idx)
        if self.infer_model_structure:
            struct_kld = self.structure_sampler.kl_divergence
            return weight_kld + struct_kld
        else:
            return weight_kld

    def reset_for_new_task(self, task_id=None):
        for layer_ in self.conv_layers:
            if self.transfer_bn_params:
                layer_.reset_for_new_task(task_idx=task_id)
            else:
                layer_.reset_for_new_task()

        for layer_ in self.fc_layers:
            layer_.reset_for_new_task()

        if self.has_bayesian_head:
            if self.single_head:
                self.out_layer[0].reset_for_new_task()
            else:
                self.out_layer[task_id].reset_for_new_task()

        if self.infer_model_structure:
            self.structure_sampler.reset_for_new_task()

        if self.use_kl_mask:
            self.union_mask, _ = self.get_union_mask()

    def generate_single_mask(self, n_samples=None):
        n_samples = self.n_test_samples if n_samples is None else n_samples
        Z_single, threshold, pi = self.structure_sampler(
            n_samples, get_single_mask=True, get_pi=True)
        model_usage = [(Z_l.sum() / Z_l.numel()).item() for Z_l in Z_single]
        return Z_single, threshold, pi, model_usage

    def add_task_mask(self, task_idx, n_samples=None):
        with torch.no_grad():
            Z_single, threshold, *_ = self.generate_single_mask(n_samples)

            for mask_store, mask in zip(self.task_mask_list[task_idx], Z_single):
                mask_store.data.copy_(mask.squeeze().detach().data)

            self.task_threshold_list[task_idx].data.copy_(threshold)

    def get_union_mask(self):
        n_tasks_completed = (self.task_threshold_list > -1).sum()
        union_mask = []
        for layer_idx in range(len(self.conv_layers) + len(self.fc_layers)):
            union_l = self.task_mask_list[0][layer_idx].data.clone()

            for task_mask in self.task_mask_list[1:n_tasks_completed]:
                union_l += task_mask[layer_idx].data.clone()
            union_l = (union_l > 0) * 1
            union_mask.append(union_l)
        return union_mask, max(self.task_threshold_list)

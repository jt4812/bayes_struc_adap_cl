import torch
from torch import nn

from src.layers.bayes_layer import BayesLinear, BayesMLPBlock
from src.models.structure_sampler import NetworkStructureSampler


class BayesAdaptiveMLP(nn.Module):
    def __init__(self, args, exp_dict=None):
        super(BayesAdaptiveMLP, self).__init__()
        self.model_args = args
        self.exp_dict = exp_dict

        self.input_feature_dim = args["input_feature_dim"]
        self.out_feature_dim = args["out_feature_dim"]

        self.max_width = args["max_width"]
        self.truncation_level = args["truncation_level"]
        self.n_layers = self.truncation_level

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.posterior_std = args["posterior_std"]

        self.device = args["device"]

        self.kl_weight = args["kl_weight"]
        if "kl_struct" in args:
            self.kl_struct = args["kl_struct"]
        self.has_norm = args["has_norm"]

        self.has_bayesian_head = args["has_bayesian_head"]
        self.use_fixed_mask = args["use_fixed_mask"]

        self.single_head = exp_dict["single_head"]
        self.n_tasks = exp_dict["n_tasks"]

        self.mask_type = args["mask_type"]  # ["neuron_mask", "weight mask"]

        self.network_arch = args["network_arch"]
        self.use_kl_mask = args["use_kl_mask"]
        self.connection_type = args["connection_type"]
        self.depth_inference = args.get("depth_inference", "z_threshold")
        self.usage_threshold = args.get("usage_threshold", 0.025)
        self.build_model()

    def build_model(self):
        # define neural network structure sampler with parameters defined in argument parser
        self.infer_model_structure = self.model_args["infer_model_structure"]
        if self.infer_model_structure:
            self.structure_sampler = NetworkStructureSampler(self.model_args)

        self.deepest_activated_layer = -1

        self.layers = nn.ModuleList(
            [BayesMLPBlock(self.input_feature_dim, self.max_width, self.posterior_std, has_norm=self.has_norm)])

        for i in range(1, self.truncation_level):
            self.layers.append(
                BayesMLPBlock(self.max_width, self.max_width, self.posterior_std, has_norm=self.has_norm,
                              residual=(self.connection_type == "skip_connection_only")))

        self.n_heads = 1 if self.single_head else self.n_tasks

        if self.has_bayesian_head:
            self.out_layer = nn.ModuleList([
                BayesLinear(self.max_width, self.out_feature_dim, posterior_std=0.001) for _ in range(self.n_heads)
            ])
        else:
            self.out_layer = nn.ModuleList([
                nn.Linear(self.max_width, self.out_feature_dim) for _ in range(self.n_heads)
            ])

        if self.use_kl_mask:
            self.union_mask = None

        if self.use_fixed_mask:
            self.task_mask_first_layer_list = nn.Parameter(
                torch.ones(self.n_tasks, self.max_width, self.input_feature_dim),
                requires_grad=False)
            self.task_mask_remaining_layer_list = nn.Parameter(
                torch.ones(self.n_tasks, self.truncation_level - 1, self.max_width, self.max_width, ),
                requires_grad=False)
            self.union_mask = None

            self.task_threshold_list = nn.Parameter(torch.ones(self.n_tasks, 1, dtype=torch.int) * -1,
                                                    requires_grad=False)

    def _forward(self, x, mask_matrix, threshold, head_idx=0):
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        if self.training and threshold > self.deepest_activated_layer:
            self.deepest_activated_layer = threshold

        h_feature = 0

        mask = mask_matrix[0]
        x = self.layers[0](x, mask)

        mask_matrix = mask_matrix[1]
        n_layers = threshold - 1

        for layer_idx in range(n_layers):
            mask = mask_matrix[layer_idx]

            if self.network_arch == "fc":
                layer_idx = layer_idx + 1

            x = self.layers[layer_idx](x, mask)

        if "relu" in self.connection_type:
            h_feature = torch.relu(h_feature)

        if self.connection_type == "skip_connection_only":
            h_feature = x

        if head_idx == -1:
            # run model to predict output from all the heads
            out = [self.out_layer[h_idx](h_feature) for h_idx in range(self.n_heads)]
            out = torch.stack(out)
        else:
            out = self.out_layer[head_idx](h_feature)
        return out

    def forward(self, x, task_idx=None, n_samples=None, use_fixed_mask=False, ret_threshold=False):
        n_samples = (self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        if self.use_fixed_mask:
            act_vec = self.forward_masked(x, task_idx, n_samples, use_fixed_mask, ret_threshold)
        else:
            act_vec = self.forward_unmasked(x, task_idx, n_samples, use_fixed_mask, ret_threshold)
        return act_vec

    def forward_unmasked(self, x, task_idx=None, n_samples=None, use_fixed_mask=False, ret_threshold=False):
        act_vec = []

        if use_fixed_mask:
            Z = self.task_mask_list[task_idx]
            threshold = self.task_threshold_list[task_idx]
        else:
            Z, threshold = self.structure_sampler(n_samples)

        head_idx = self.get_head_idx(task_idx)

        for s in range(n_samples):
            Z_mask = Z if use_fixed_mask else (Z[0][s], Z[1][s])

            out = self._forward(
                x, Z_mask, threshold, head_idx,
            )
            act_vec.append(out.unsqueeze(0))

        act_vec = torch.cat(act_vec, dim=0)
        if ret_threshold:
            return act_vec, threshold
        else:
            return act_vec

    def forward_masked(self, x, task_idx=None, n_samples=None, use_fixed_mask=False, ret_threshold=False):
        act_vec = []

        if not use_fixed_mask:
            Z, threshold = self.structure_sampler(
                n_samples)

        task_range = [task_idx]

        for s in range(n_samples):
            if not use_fixed_mask:
                Z_mask = (Z[0][s], Z[1][s])

            act_vec_sample = []

            for task_idx_ in task_range:
                if use_fixed_mask:
                    Z_mask = (
                        self.task_mask_first_layer_list[task_idx_], self.task_mask_remaining_layer_list[task_idx_])

                    threshold = self.task_threshold_list[task_idx_]

                out = self._forward(
                    x, Z_mask, threshold, head_idx=task_idx_
                )
                act_vec_sample.append(out.unsqueeze(0))

            act_vec_sample = torch.cat(act_vec_sample, dim=0)
            act_vec.append(act_vec_sample.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)  # n_samples x tasks_count x batch_size x n_classes

        # squeeze first dim if only one task
        act_vec = act_vec.squeeze(dim=1)

        if ret_threshold:
            return act_vec, threshold
        else:
            return act_vec

    def get_E_loglike(self, neg_loglike_fun, y_preds, y_true):
        n_samples = self.n_train_samples

        batch_size = len(y_true)
        y_true = y_true.squeeze().repeat(n_samples)
        y_preds = y_preds.view(n_samples * batch_size, -1)

        neg_loglike = neg_loglike_fun(y_preds, y_true)
        E_neg_loglike = neg_loglike.mean()

        return E_neg_loglike

    def estimate_total_loss(self, neg_loglike_fun, y_preds, y_true, kl_scale=1, use_fixed_mask=False, task_idx=None):
        E_neg_loglike = self.get_E_loglike(neg_loglike_fun, y_preds, y_true)

        weight_kld = self.kl_divergence_weight(task_idx) * self.kl_weight
        kl_struct = self.kl_struct if hasattr(self, "kl_struct") else self.kl_weight

        struct_kld = self.structure_sampler.kl_divergence * kl_struct

        if use_fixed_mask:
            kld = weight_kld
        else:
            kld = weight_kld + struct_kld

        # KL term weighted by kl_scale for ELBO consistency in mini batch and kl_weight for regularization effectiveness
        total_loss = E_neg_loglike + kld * kl_scale
        return total_loss, E_neg_loglike, weight_kld * kl_scale, struct_kld * kl_scale

    def get_head_idx(self, task_idx):
        if self.single_head:
            head_idx = 0  # use head at idx 0 irrespective of cl type
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
            (union_mask_first_layer, union_mask_remaining_layers) = self.union_mask
            weight_kld += self.layers[0].masked_kl_divergence(union_mask_first_layer)

            for layer_idx in range(1, self.deepest_activated_layer):
                weight_kld += self.layers[layer_idx].masked_kl_divergence(union_mask_remaining_layers[layer_idx - 1])
        else:
            for layer_ in self.layers[:self.deepest_activated_layer]:
                weight_kld += layer_.kl_divergence

        if self.has_bayesian_head:
            if self.single_head:
                weight_kld += self.out_layer[0].kl_divergence
            else:
                weight_kld += self.out_layer[task_idx].kl_divergence

        return weight_kld

    def kl_divergence(self, task_idx=None):
        weight_kld = self.kl_divergence_weight(task_idx)
        struct_kld = self.structure_sampler.kl_divergence
        return weight_kld + struct_kld

    def reset_for_new_task(self, task_id=None):
        for layer_ in self.layers:
            layer_.reset_for_new_task()

        if self.has_bayesian_head:
            if self.single_head:
                self.out_layer[0].reset_for_new_task()
            else:
                self.out_layer[task_id].reset_for_new_task()

        self.structure_sampler.reset_for_new_task()

        if self.use_kl_mask:
            self.union_mask, _ = self.get_union_mask()

    def generate_single_mask(self, n_samples=None):
        n_samples = self.n_test_samples if n_samples is None else n_samples
        Z_single, threshold, pi = self.structure_sampler(n_samples, get_single_mask=True, get_pi=True)
        model_usage = torch.tensor(
            [(Z_single[0].sum() / Z_single[0].numel()).item()] + [(Z_l.sum() / Z_l.numel()).item() for Z_l in
                                                                  Z_single[1].squeeze(0)])
        if self.depth_inference == "usage_threshold":
            threshold_new = (model_usage > self.usage_threshold).sum().item()
            for Z_l in Z_single[1][:, threshold_new - 1:]:
                Z_l.data.zero_()

            model_usage = torch.tensor(
                [(Z_single[0].sum() / Z_single[0].numel()).item()] + [(Z_l.sum() / Z_l.numel()).item() for Z_l in
                                                                      Z_single[1].squeeze(0)])
            return Z_single, threshold_new, pi, model_usage

        else:
            return Z_single, threshold, pi, model_usage

    def add_task_mask(self, task_idx, n_samples=None):
        with torch.no_grad():
            Z_single, threshold, *_ = self.generate_single_mask(n_samples)

            self.task_mask_first_layer_list[task_idx].data.copy_(Z_single[0].squeeze().detach().data)
            self.task_mask_remaining_layer_list[task_idx].data.copy_(Z_single[1].squeeze().detach().data)

            self.task_threshold_list[task_idx].data.copy_(threshold)

    def get_union_mask(self, n_tasks_to_union=None):
        if n_tasks_to_union is None:
            n_tasks_completed = (self.task_threshold_list > -1).sum()
        else:
            n_tasks_completed = n_tasks_to_union

        union_mask_first_layer = ((self.task_mask_first_layer_list[:n_tasks_completed].sum(dim=0)) > 0) * 1
        union_mask_remaining_layers = ((self.task_mask_remaining_layer_list[:n_tasks_completed].sum(dim=0)) > 0) * 1
        union_mask = (union_mask_first_layer, union_mask_remaining_layers)

        return union_mask, max(self.task_threshold_list)

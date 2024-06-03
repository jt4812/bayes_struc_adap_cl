import torch
from torch import nn

from src.layers.layers_vae import BayesMLPBlock, BayesLinear
from src.vae.models.structure_sampler_vae import NetworkStructureSampler


class BayesAdaptiveDecoder(nn.Module):
    def __init__(self, args, exp_dict=None):
        super(BayesAdaptiveDecoder, self).__init__()

        self.input_feature_dim = args["input_feature_dim"]
        self.latent_feature_dim = args["latent_feature_dim"]
        self.max_width = args["max_width"]

        self.n_head_layers = args["n_head_layers"]
        self.n_shared_dec_layers = args["n_shared_dec_layers"]
        self.truncation_level = self.n_shared_dec_layers

        self.heads = nn.ModuleList(
            [self.create_head(self.latent_feature_dim, self.max_width, self.n_head_layers) for _ in
             range(exp_dict["n_tasks"])])

        self.shared_decoder = nn.ModuleList(
            [BayesMLPBlock(self.max_width, self.max_width, act_layer_fn="relu", residual=True)
             for _ in range(self.n_shared_dec_layers)])

        self.final_layer_decoder = BayesLinear(self.max_width, self.input_feature_dim)

        args["truncation_level"] = self.n_shared_dec_layers
        self.structure_sampler = NetworkStructureSampler(args)

    def create_head(self, latent_feature_dim, max_width, n_head_layers):
        model_size = [latent_feature_dim, ] + [max_width] * n_head_layers
        head_layers = []
        for layer_idx in range(n_head_layers):
            prev_layer_width = model_size[layer_idx]
            next_layer_width = model_size[layer_idx + 1]

            layer_ = BayesMLPBlock(prev_layer_width, next_layer_width, act_layer_fn="relu")
            head_layers.append(layer_)
        return nn.Sequential(*head_layers)

    def forward_backbone(self, x, sample_W=True):
        mask_matrix, threshold = self.structure_sampler()
        mask_matrix = mask_matrix.squeeze(dim=0)  # K x M x M or K x M

        if self.training:
            self.deepest_activated_layer = threshold

        for layer_idx in range(threshold):
            x = self.shared_decoder[layer_idx](x, mask_matrix[layer_idx], sample_W=sample_W)

        out = self.final_layer_decoder(x, sample_W=sample_W)
        return out

    def forward(self, z, task_idx, sample_W=True):
        h = self.heads[task_idx](z)
        x_ = torch.sigmoid(self.forward_backbone(h, sample_W=sample_W))
        return x_

    def generate_samples(self, task_idx, n_samples=100):
        z = torch.randn((n_samples, self.latent_feature_dim)).cuda()
        h = self.heads[task_idx](z)
        x_ = torch.sigmoid(self.forward_backbone(h))
        return x_.detach().cpu()

    def kl_divergence_weight(self, task_id, regularize_head=False):
        weight_kld = 0
        if regularize_head:
            weight_kld = self.heads[task_id][0].kl_divergence

        for layer_ in self.shared_decoder:
            weight_kld += layer_.kl_divergence

        weight_kld += self.final_layer_decoder.kl_divergence

        return weight_kld

    # Done
    def kl_divergence_struc(self):
        return self.structure_sampler.kl_divergence

    # Done
    def reset_for_new_task(self):
        for layer_ in self.shared_decoder:
            layer_.reset_for_new_task()

        self.final_layer_decoder.reset_for_new_task()

        self.structure_sampler.reset_for_new_task()

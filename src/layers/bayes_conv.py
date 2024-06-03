import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


class BayesConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, posterior_std=0.001,
                 share_std_channel=False):
        super(BayesConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        std_shape = (out_channels, in_channels) if share_std_channel else (out_channels, in_channels, *kernel)

        self.p_weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel), requires_grad=False)
        self.p_weight_sig = nn.Parameter(torch.ones(*std_shape), requires_grad=False)
        self.p_bias_mu = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.p_bias_sig = nn.Parameter(torch.ones(out_channels), requires_grad=False)

        self.q_weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, *kernel))
        self.q_weight_sig = nn.Parameter(torch.empty(*std_shape))
        self.q_bias_mu = nn.Parameter(torch.empty(out_channels))
        self.q_bias_sig = nn.Parameter(torch.empty(out_channels))

        self.eps = 1e-8
        self.initialize_posterior_params(posterior_std)

    def initialize_posterior_params(self, posterior_std):
        posterior_std_proxy = np.log(np.exp(posterior_std) - 1)
        nn.init.uniform_(self.q_weight_mu, -0.2, 0.2)
        nn.init.constant_(self.q_weight_sig, posterior_std_proxy)
        nn.init.constant_(self.q_bias_mu, 0)
        nn.init.constant_(self.q_bias_sig, posterior_std_proxy)

    @property
    def posterior_dist(self):
        q_weight_sig = self.q_weight_sig
        if len(self.q_weight_sig.shape) == 2:
            q_weight_sig = self.q_weight_sig.unsqueeze(-1).unsqueeze(-1).expand(
                self.out_channels, self.in_channels, *self.kernel)
        q_weight = Normal(self.q_weight_mu, F.softplus(q_weight_sig) + self.eps)
        q_bias = Normal(self.q_bias_mu, F.softplus(self.q_bias_sig) + self.eps)
        return q_weight, q_bias

    @property
    def prior_dist(self):
        p_weight_sig = self.p_weight_sig
        if len(self.q_weight_sig.shape) == 2:
            p_weight_sig = self.p_weight_sig.unsqueeze(-1).unsqueeze(-1).expand(
                self.out_channels, self.in_channels, *self.kernel)
        p_weight = Normal(self.p_weight_mu, F.softplus(p_weight_sig) + self.eps)
        p_bias = Normal(self.p_bias_mu, F.softplus(self.p_bias_sig) + self.eps)
        return p_weight, p_bias

    def sample_weight(self):
        q_weight, q_bias = self.posterior_dist
        weight = q_weight.rsample()
        bias = q_bias.rsample()
        return weight, bias

    def forward(self, x, sample=True, mask=None):
        weight, bias = self.sample_weight() if sample else (self.weight_mu, self.bias_mu)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            weight = weight * mask
        return F.conv2d(x, weight, bias, padding=self.padding, stride=self.stride)

    @property
    def kl_divergence(self):
        q_weight, q_bias = self.posterior_dist
        p_weight, p_bias = self.prior_dist
        return kl_divergence(q_weight, p_weight).sum() + kl_divergence(q_bias, p_bias).sum()

    def masked_kl_divergence(self, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1).unsqueeze(-1)

        p_weight_sig = self.p_weight_sig
        q_weight_sig = self.q_weight_sig

        if len(self.q_weight_sig.shape) == 2:
            p_weight_sig = self.p_weight_sig.unsqueeze(-1).unsqueeze(-1).expand(
                self.out_channels, self.in_channels, *self.kernel)
            q_weight_sig = self.q_weight_sig.unsqueeze(-1).unsqueeze(-1).expand(
                self.out_channels, self.in_channels, *self.kernel)

        q_weight_mu = self.q_weight_mu * mask
        q_weight_sig_ = F.softplus(q_weight_sig) + self.eps
        q_weight_sig = q_weight_sig_ * mask + (1 - mask)

        q_weight = Normal(q_weight_mu, q_weight_sig)
        q_bias = Normal(self.q_bias_mu, F.softplus(self.q_bias_sig) + self.eps)

        p_weight_mu = self.p_weight_mu * mask
        p_weight_sig_ = F.softplus(p_weight_sig) + self.eps
        p_weight_sig = p_weight_sig_ * mask + (1 - mask)

        p_weight = Normal(p_weight_mu, p_weight_sig)
        p_bias = Normal(self.p_bias_mu, F.softplus(self.p_bias_sig) + self.eps)

        return kl_divergence(q_weight, p_weight).sum() + kl_divergence(q_bias, p_bias).sum()

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, kernel={}, stride={} bias={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride, self.q_bias_mu is not None)

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
        if mask is None:
            self.p_weight_mu.data.copy_(self.q_weight_mu.data)
            self.p_weight_sig.data.copy_(self.q_weight_sig.data)
            self.p_bias_mu.data.copy_(self.q_bias_mu.data)
            self.p_bias_sig.data.copy_(self.q_bias_sig.data)

        elif mask_type == "neuron_mask":
            for active_idx in torch.where(mask == 1):
                self.p_weight_mu[active_idx].data.copy_(self.q_weight_mu[active_idx].data)
                self.p_weight_sig[active_idx].data.copy_(self.q_weight_sig[active_idx].data)
                self.p_bias_mu[active_idx].data.copy_(self.q_bias_mu[active_idx].data)
                self.p_bias_sig[active_idx].data.copy_(self.q_bias_sig[active_idx].data)

        elif mask_type == "weight_mask":
            self.p_weight_mu.data.copy_(mask * self.q_weight_mu.data + (1 - mask) * self.p_weight_mu.data)
            self.p_weight_sig.data.copy_(mask * self.q_weight_sig.data + (1 - mask) * self.p_weight_sig.data)
            # self.p_bias_mu.data.copy_(mask * self.q_bias_mu.data + (1 - mask) * self.p_bias_mu.data)
            # self.p_bias_sig.data.copy_(mask * self.q_bias_sig.data + (1 - mask) * self.p_bias_sig.data)
            self.p_bias_mu.data.copy_(self.q_bias_mu.data)
            self.p_bias_sig.data.copy_(self.q_bias_sig.data)


    def get_snr(self, type_="snr"):
        if type_ == "snr":
            snr_weight = torch.abs(self.q_weight_mu) / (F.softplus(self.q_weight_sig) + self.eps)
        elif type_ == "precision":
            snr_weight = 1 / (F.softplus(self.q_weight_sig) + self.eps)
        return snr_weight  # , snr_bias

    def get_weight_shape(self):
        return (self.out_channels, self.in_channels, *self.kernel)


class BayesConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, posterior_std=0.001, has_norm=False,
                 has_max_pool=True, share_std_channel=False,
                 residual=False, use_task_specific_norms=False, n_tasks=None):
        super(BayesConvBlock, self).__init__()

        self.conv = BayesConv(in_channels, out_channels, kernel, (1, 1) if has_max_pool else stride, padding,
                              posterior_std, share_std_channel)
        self.act_layer_fn = nn.LeakyReLU()

        self.has_norm = has_norm
        if has_norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)

            self.use_task_specific_norms = use_task_specific_norms
            if use_task_specific_norms:
                self.n_tasks = n_tasks
                self.norm_layer_ls = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(n_tasks)])
        self.has_max_pool = has_max_pool
        if has_max_pool:
            self.max_layer = nn.MaxPool2d(stride)
        self.residual = residual

        self.share_std_channel = share_std_channel

    def forward(self, x, mask=None, task_idx=None):
        residual = x

        out = self.conv(x, mask=mask)

        if self.has_norm:
            if task_idx is None:
                out = self.norm_layer(out)
            else:
                out = self.norm_layer_ls[task_idx](out)

        out = self.act_layer_fn(out)

        if self.has_max_pool:
            out = self.max_layer(out)

        if self.residual:
            out += residual

        return out

    @property
    def kl_divergence(self):
        return self.conv.kl_divergence

    def masked_kl_divergence(self, mask):
        return self.conv.masked_kl_divergence(mask)

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask", task_idx=None):
        self.conv.reset_for_new_task(mask, mask_type)

        if task_idx is not None and self.use_task_specific_norms:
            if task_idx < self.n_tasks - 1:
                self.norm_layer_ls[task_idx + 1].load_state_dict(self.norm_layer_ls[task_idx].state_dict())

    def get_snr(self, type_):
        return self.conv.get_snr(type_)

    def get_weight_shape(self):
        return self.conv.get_weight_shape()


class NPBayesConvSuperBlock(nn.Module):
    def __init__(self, n_channels, posterior_std, has_norm, truncation_level):
        super(NPBayesConvSuperBlock, self).__init__()

        self.block_layers = nn.ModuleList([
            BayesConvBlock(n_channels, n_channels, (3, 3), (1, 1), posterior_std=posterior_std, has_norm=has_norm,
                           residual=True) for _ in
            range(truncation_level)
        ])
        self.deepest_activated_layer = -1

    def forward(self, x, mask_matrix, threshold):
        if self.training and threshold > self.deepest_activated_layer:
            self.deepest_activated_layer = threshold
        for layer_idx in range(threshold):
            _layer = self.block_layers[layer_idx]
            mask = mask_matrix[layer_idx]
            x = _layer(x, mask)
        return x

    @property
    def kl_divergence(self):
        kld = 0
        for _layer in self.block_layers[:self.deepest_activated_layer]:
            kld += _layer.kl_divergence
        return kld

    def reset_for_new_task(self, mask_ls=None):
        for _layer in self.block_layers:
            _layer.reset_for_new_task()

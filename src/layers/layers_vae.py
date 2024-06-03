import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_weight_mu = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.p_weight_log_sig = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.p_bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.p_bias_log_sig = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.q_weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.q_weight_log_sig = nn.Parameter(torch.empty(out_features, in_features))
        self.q_bias_mu = nn.Parameter(torch.empty(out_features))
        self.q_bias_log_sig = nn.Parameter(torch.empty(out_features))

        self.initialize_posterior_params()

    def initialize_posterior_params(self):
        nn.init.xavier_uniform_(self.q_weight_mu)
        nn.init.constant_(self.q_weight_log_sig, -6)
        nn.init.constant_(self.q_bias_mu, 0)
        nn.init.constant_(self.q_bias_log_sig, -6)

    @property
    def posterior_dist(self):
        q_weight = Normal(self.q_weight_mu, self.q_weight_log_sig.exp())
        q_bias = Normal(self.q_bias_mu, self.q_bias_log_sig.exp())
        return q_weight, q_bias

    @property
    def prior_dist(self):
        p_weight = Normal(self.p_weight_mu, self.p_weight_log_sig.exp())
        p_bias = Normal(self.p_bias_mu, self.p_bias_log_sig.exp())
        return p_weight, p_bias

    def sample_weight(self):
        q_weight, q_bias = self.posterior_dist
        weight = q_weight.rsample()
        bias = q_bias.rsample()
        return weight, bias

    def forward(self, x, sample_W=True, mask=None):
        weight, bias = self.sample_weight() if sample_W else (self.q_weight_mu, self.q_bias_mu)
        if mask is not None:
            weight = weight * mask
            bias = bias * mask.max(dim=1)[0]
        return F.linear(x, weight, bias)

    @property
    def kl_divergence(self):
        q_weight, q_bias = self.posterior_dist
        p_weight, p_bias = self.prior_dist
        return kl_divergence(q_weight, p_weight).sum() + kl_divergence(q_bias, p_bias).sum()

    def masked_kl_divergence(self, mask):
        q_weight_mu = self.q_weight_mu * mask
        q_weight_log_sig_ = self.q_weight_log_sig
        q_weight_log_sig = q_weight_log_sig_ * mask + (1 - mask)

        q_weight = Normal(q_weight_mu, q_weight_log_sig)
        q_bias = Normal(self.q_bias_mu, self.q_weight_log_sig)

        p_weight_mu = self.p_weight_mu * mask
        p_weight_log_sig_ = self.p_weight_log_sig
        p_weight_log_sig = p_weight_log_sig_ * mask + (1 - mask)

        p_weight = Normal(p_weight_mu, p_weight_log_sig)
        p_bias = Normal(self.p_bias_mu, self.p_bias_log_sig)

        return kl_divergence(q_weight, p_weight).sum() + kl_divergence(q_bias, p_bias).sum()

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
        if mask is None:
            self.p_weight_mu.data.copy_(self.q_weight_mu.data)
            self.p_weight_log_sig.data.copy_(self.q_weight_log_sig.data)
            self.p_bias_mu.data.copy_(self.q_bias_mu.data)
            self.p_bias_log_sig.data.copy_(self.q_bias_log_sig.data)

        elif mask_type == "neuron_mask":
            for active_idx in torch.where(mask == 1):
                self.p_weight_mu[active_idx].data.copy_(self.q_weight_mu[active_idx].data)
                self.p_weight_log_sig[active_idx].data.copy_(self.q_weight_log_sig[active_idx].data)
                self.p_bias_mu[active_idx].data.copy_(self.q_bias_mu[active_idx].data)
                self.p_bias_log_sig[active_idx].data.copy_(self.q_bias_log_sig[active_idx].data)

        elif mask_type == "weight_mask":
            self.p_weight_mu.data.copy_(mask * self.q_weight_mu.data + (1 - mask) * self.p_weight_mu.data)
            self.p_weight_log_sig.data.copy_(
                mask * self.q_weight_log_sig.data + (1 - mask) * self.p_weight_log_sig.data)
            self.p_bias_mu.data.copy_(self.q_bias_mu.data)
            self.p_bias_log_sig.data.copy_(self.q_bias_log_sig.data)

        self.q_weight_log_sig.data.copy_(torch.ones_like(self.q_weight_log_sig.data) * -6)
        self.q_bias_log_sig.data.copy_(torch.ones_like(self.q_bias_log_sig.data) * -6)

    def get_snr(self, eps=1e-6):
        snr_weight = torch.abs(self.q_weight_mu) / (self.q_weight_sig.exp() + eps)
        # snr_bias = torch.abs(self.q_bias_mu) / (self.q_bias_log_sig.exp() + eps)
        return snr_weight  # , snr_bias

class BayesMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, has_norm=False, residual=False,
                 act_layer_fn=None):
        super(BayesMLPBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = BayesLinear(in_features, out_features)
        if act_layer_fn == "sigmoid":
            self.act_layer_fn = nn.Sigmoid()
        elif act_layer_fn == "relu":
            self.act_layer_fn = nn.ReLU(inplace=False)
        else:
            self.act_layer_fn = nn.LeakyReLU(inplace=False)

        self.has_norm = has_norm
        if has_norm:
            self.norm_layer = nn.BatchNorm1d(out_features)
        self.residual = residual

    def forward(self, x, mask=None, sample_W=True):
        residual = x

        if mask is not None:
            assert len(mask.shape) in [1, 2]
            use_neuron_mask = (len(mask.shape) == 1)

        if (mask is not None) and use_neuron_mask:
            out = self.linear(x, sample_W=sample_W)
        else:
            out = self.linear(x, mask=mask, sample_W=sample_W)

        if self.has_norm:
            out = self.norm_layer(out)

        out = self.act_layer_fn(out)

        if (mask is not None) and use_neuron_mask:
            out *= mask.view(1, -1)

        if self.residual:
            out = out + residual

        return out

    @property
    def kl_divergence(self):
        return self.linear.kl_divergence

    def masked_kl_divergence(self, mask):
        return self.linear.masked_kl_divergence(mask)

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
        self.linear.reset_for_new_task(mask, mask_type)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, has_norm={}, has_residual={}'.format(
            self.in_features, self.out_features, self.has_norm, self.residual)

    def get_snr(self):
        return self.linear.get_snr()

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Beta, RelaxedBernoulli
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class NetworkStructureSampler(nn.Module):
    def __init__(self, args):
        super(NetworkStructureSampler, self).__init__()

        self.args = args
        # epsilon to select the number of layers with activated neurons
        self.act_thresh = args["act_thresh"]

        self.max_width = args["max_width"] if "max_width" in args else args["n_channels"]
        self.truncation_level = args["truncation_level"]
        # Temperature for Concrete Bernoulli
        self.prior_temperature = torch.tensor(args["prior_temperature"])
        self.posterior_temperature = torch.tensor(args["posterior_temperature"])

        self.network_arch = args["network_arch"]

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.mask_type = args["mask_type"]  # "weight mask"

        self.input_feature_dim = args["input_feature_dim"]

        # inverse softplus to avoid parameter underflow
        alpha = np.log(np.exp(args["a_prior"]) - 1)
        beta = np.log(np.exp(args["b_prior"]) - 1)

        # Hyper-parameters for prior beta
        self.alpha = nn.Parameter(torch.full((self.truncation_level,), alpha), requires_grad=False)
        self.beta = nn.Parameter(torch.full((self.truncation_level,), beta), requires_grad=False)

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.full((self.truncation_level,), alpha))
        self.b_k = nn.Parameter(torch.full((self.truncation_level,), beta))

        self.eps = 0.01

        # accumulates sample based kl divergence between concrete bernoullis
        self.kld_bernoulli = 0

        self.use_sigma_k = args["use_sigma_k"]
        self.sigma_k_init = 0.5

        if self.use_sigma_k:
            sigma_k_first_layer_shape = [self.max_width, self.input_feature_dim]
            sigma_k_remaining_layer_shape = [self.truncation_level - 1, self.max_width, self.max_width]
            self.sigma_k = nn.ParameterList([
                nn.Parameter(torch.logit(torch.ones(sigma_k_first_layer_shape) * self.sigma_k_init),
                             requires_grad=True),
                nn.Parameter(torch.logit(torch.ones(sigma_k_remaining_layer_shape) * self.sigma_k_init),
                             requires_grad=True)
            ])

        self.pi_expander = self.expand_pi_

    @property
    def prior_beta(self):
        alpha = F.softplus(self.alpha) + self.eps
        beta = F.softplus(self.beta) + self.eps
        return Beta(alpha, beta)

    @property
    def variational_beta(self):
        a_k = F.softplus(self.a_k) + self.eps
        b_k = F.softplus(self.b_k) + self.eps
        return Beta(a_k, b_k)

    @property
    def variational_params(self):
        a_k = F.softplus(self.a_k) + self.eps
        b_k = F.softplus(self.b_k) + self.eps
        return a_k, b_k

    def get_threshold(self, Z):
        Z = Z[1]
        # First, count the number of neurons in each layer
        # Z shape: (n_samples, K, M, M)
        threshold_Z = (Z > self.act_thresh).sum(dim=[2, 3])

        assert len(threshold_Z.shape) == 2
        # Second, compute the layers with activated neurons
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        # Third, consider maximum of thresholds from multiple samples
        threshold = max(threshold_array)
        # increase threshold by one for weight mask to compensate the removal of first layer.
        return threshold + 1

    @property
    def expand_pi_(self):
        # x: shape (n_samples, K)
        return lambda \
                x: (x[:, 0][[..., ] + [None, ] * 2].expand(-1, self.max_width, self.input_feature_dim),
                    x[:, 1:][[..., ] + [None, ] * 2].expand(-1, -1, self.max_width, self.max_width))

    def forward(self, n_samples=None, get_pi=False, get_single_mask=False):
        n_samples = (
            self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        # sample from variational beta distribution
        ν = self.variational_beta.rsample((n_samples,)).view(n_samples, self.truncation_level)  # n_samples x K
        pi = torch.cumsum(ν.log(), dim=1).exp()  # n_samples x K

        pi_expanded = self.pi_expander(pi)  # n_samples x K x M or n_samples x K x M x M

        if self.use_sigma_k:
            pi_expanded = self.add_sigma_k_pi_expanded(pi_expanded)

        if pi_expanded[0].isnan().any() and pi_expanded[1].isnan().any():
            print(pi_expanded[0].isnan().any())
            print(pi_expanded[1].isnan().any())

        # sample binary mask z_l given the activation level π_l of the layer
        Z = self.sample_mask(pi_expanded, get_single_mask, pi)

        threshold = self.get_threshold(Z)

        if get_pi:
            keep_prob = pi.detach().mean(0)
            return Z, threshold, keep_prob

        return Z, threshold

    def add_sigma_k_pi_expanded(self, pi_expanded):
        sigma_k_first_layer, sigma_k_remaining_layer = self.sigma_k
        pi_expanded_first_layer, pi_expanded_remaining_layer = pi_expanded

        logit_first_layer = torch.logit(pi_expanded_first_layer)
        logit_remaining_layer = torch.logit(pi_expanded_remaining_layer)

        logit_first_layer_ = logit_first_layer + sigma_k_first_layer.unsqueeze(0)
        logit_remaining_layer_ = logit_remaining_layer + sigma_k_remaining_layer.unsqueeze(0)

        pi_first_layer = torch.sigmoid(logit_first_layer_)
        pi_remaining_layer = torch.sigmoid(logit_remaining_layer_)
        pi_expanded = (
            pi_first_layer,
            pi_remaining_layer
        )

        return pi_expanded

    def sample_mask(self, pi_expanded, get_single_mask, pi):
        Z = self.sample_weight_mask(pi_expanded, get_single_mask, self.pi_expander(pi))
        return Z

    def sample_weight_mask(self, pi, get_single_mask=False, pi_prior=None):
        if self.training:
            concrete_bernoulli_dis_first_layer = RelaxedBernoulli(probs=pi[0], temperature=self.posterior_temperature)
            concrete_bernoulli_dist_remaining = RelaxedBernoulli(probs=pi[1], temperature=self.posterior_temperature)

            Z_first_layer = concrete_bernoulli_dis_first_layer.rsample()
            Z_remaining = concrete_bernoulli_dist_remaining.rsample()

            if self.use_sigma_k:
                self.kld_bernoulli = self.calculate_kl_divergence_bernoulli(pi[0], pi_prior[0], Z_first_layer) + \
                                     self.calculate_kl_divergence_bernoulli(pi[1], pi_prior[1], Z_remaining)

        elif get_single_mask:
            # calculates mean of the pis and generates single mask
            pi_first_layer = pi[0].mean(0)
            pi_remaining = pi[1].mean(0)

            bernoulli_dis_first_layer = Bernoulli(probs=pi_first_layer)
            bernoulli_dist_remaining = Bernoulli(probs=pi_remaining)

            Z_first_layer = bernoulli_dis_first_layer.sample().unsqueeze(dim=0)
            Z_remaining = bernoulli_dist_remaining.sample().unsqueeze(dim=0)
        else:
            bernoulli_dis_first_layer = Bernoulli(probs=pi[0])
            bernoulli_dist_remaining = Bernoulli(probs=pi[1])

            Z_first_layer = bernoulli_dis_first_layer.sample()
            Z_remaining = bernoulli_dist_remaining.sample()

        Z = [Z_first_layer, Z_remaining]
        return Z

    @property
    def kl_divergence(self):
        kl_beta = kl_divergence(self.variational_beta, self.prior_beta).sum()
        kl_bernoulli = self.kld_bernoulli
        return kl_beta + kl_bernoulli

    def calculate_kl_divergence_bernoulli(self, posterior_pi, prior_pi, Z):
        posterior_dist = RelaxedBernoulli(probs=posterior_pi, temperature=self.posterior_temperature)
        prior_dist = RelaxedBernoulli(probs=prior_pi, temperature=self.prior_temperature)

        log_prob_posterior = posterior_dist.log_prob(Z)
        log_prob_prior = prior_dist.log_prob(Z)

        kld = (log_prob_posterior - log_prob_prior)  # S x K x M, or S x K x M x M
        return kld.mean(0).sum()

    def reset_for_new_task(self):
        self.alpha.data.copy_(self.a_k.data)
        self.beta.data.copy_(self.b_k.data)

        if self.use_sigma_k:
            self.sigma_k[0].data.copy_(torch.ones_like(self.sigma_k[0]) * self.sigma_k_init)
            self.sigma_k[1].data.copy_(torch.ones_like(self.sigma_k[1]) * self.sigma_k_init)

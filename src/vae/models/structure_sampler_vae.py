import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Beta, RelaxedBernoulli
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class NetworkStructureSampler(nn.Module):
    def __init__(self, args, eps=0.01):
        super(NetworkStructureSampler, self).__init__()
        self.args = args

        # epsilon to select the number of layers with activated neurons
        self.act_thresh = args["act_thresh"]

        self.max_width = args["max_width"]
        self.truncation_level = args["truncation_level"]
        # Temperature for Concrete Bernoulli
        self.prior_temperature = torch.tensor(args["prior_temperature"])
        self.posterior_temperature = torch.tensor(args["posterior_temperature"])

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.mask_type = args["mask_type"]  # ["neuron_mask", "weight_mask"]

        # inverse softplus to avoid parameter underflow
        alpha = np.log(np.exp(args["a_prior"]) - 1)
        beta = np.log(np.exp(args["b_prior"]) - 1)

        # Hyper-parameters for prior beta
        self.alpha = nn.Parameter(torch.full((self.truncation_level,), alpha), requires_grad=False)
        self.beta = nn.Parameter(torch.full((self.truncation_level,), beta), requires_grad=False)

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.full((self.truncation_level,), alpha))
        self.b_k = nn.Parameter(torch.full((self.truncation_level,), beta))

        self.eps = eps

        # accumulates sample based kl divergence between concrete Bernoulli's
        self.kld_bernoulli = 0

        self.use_sigma_k = args["use_sigma_k"]
        if self.use_sigma_k:
            sigma_k_shape = [self.truncation_level, self.max_width]
            if self.mask_type == "weight_mask":
                sigma_k_shape += [self.max_width, ]

            self.sigma_k = nn.Parameter(torch.logit(torch.ones(sigma_k_shape) * 0.5), requires_grad=True)

        self.use_prior_for_kld_ber = False
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

    def get_threshold(self, Z):
        if self.mask_type == "neuron_mask":
            # Z shape: (n_samples, K, M)
            threshold_Z = (Z > self.act_thresh).sum(2)
        elif self.mask_type == "weight_mask":
            # Z shape: (n_samples, K, M, M)
            threshold_Z = (Z > self.act_thresh).sum(dim=[2, 3])

        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        threshold = max(threshold_array)
        # assert threshold < 30
        return threshold

    @property
    def expand_pi_(self):
        if self.mask_type == "neuron_mask":
            return lambda x: x.unsqueeze(-1).expand(-1, -1, self.max_width)

        elif self.mask_type == "weight_mask":
            return lambda x: x[[..., ] + [None, ] * 2].expand(-1, -1, self.max_width, self.max_width)

    def forward(self, n_samples=1, get_pi=False, get_single_mask=False):
        # n_samples = (
        #     self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        # sample from variational beta distribution
        ν = self.variational_beta.rsample((n_samples,)).view(n_samples, self.truncation_level)  # n_samples x K
        pi = torch.cumsum(ν.log(), dim=1).exp()  # n_samples x K

        pi_expanded = self.pi_expander(pi)  # n_samples x K x M or n_samples x K x M x M

        if self.use_sigma_k:
            pi_expanded = torch.sigmoid(self.sigma_k.unsqueeze(dim=0) + torch.logit(pi_expanded))

        if self.training:
            concrete_bernoulli_dist = RelaxedBernoulli(probs=pi_expanded, temperature=self.posterior_temperature)
            Z = concrete_bernoulli_dist.rsample()

            # if self.use_sigma_k:
            #     self.calculate_kl_divergence_bernoulli(pi, pi_prior, Z)

        else:
            bernoulli_dist = Bernoulli(probs=pi_expanded.mean(0) if get_single_mask else pi_expanded)
            Z = bernoulli_dist.sample()
            Z = torch.unsqueeze(Z, dim=0) if get_single_mask else Z

        threshold = self.get_threshold(Z)

        if get_pi:
            keep_prob = pi.detach().mean(0)
            return Z, threshold, keep_prob

        return Z, threshold

    # Done
    @property
    def kl_divergence(self):
        kl_beta = kl_divergence(self.variational_beta, self.prior_beta).sum()
        kl_bernoulli = self.kld_bernoulli
        return kl_beta + kl_bernoulli

    # TODO: Implement for sigma_k case
    def calculate_kl_divergence_bernoulli(self, posterior_pi, prior_pi, Z):
        posterior_dist = RelaxedBernoulli(probs=posterior_pi, temperature=self.posterior_temperature)
        prior_dist = RelaxedBernoulli(probs=prior_pi, temperature=self.prior_temperature)

        log_prob_posterior = posterior_dist.log_prob(Z)
        log_prob_prior = prior_dist.log_prob(Z)

        kld = (log_prob_posterior - log_prob_prior)  # S x K x M, or S x K x M x M
        self.kld_bernoulli = kld.mean(0).sum()

    # Done
    def reset_for_new_task(self):
        self.alpha.data.copy_(self.a_k.data)
        self.beta.data.copy_(self.b_k.data)

        if self.use_sigma_k:
            self.sigma_k.data.copy_(torch.ones_like(self.sigma_k) * 0.5)

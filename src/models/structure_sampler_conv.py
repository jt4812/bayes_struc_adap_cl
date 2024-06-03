import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Beta, RelaxedBernoulli
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class NetworkStructureSampler(nn.Module):
    def __init__(self, args, weight_shape_ls=None):
        super(NetworkStructureSampler, self).__init__()

        self.args = args
        # epsilon to select the number of layers with activated neurons
        self.act_thresh = args["act_thresh"]

        # Temperature for Concrete Bernoulli
        self.prior_temperature = torch.tensor(args["prior_temperature"])
        self.posterior_temperature = torch.tensor(args["posterior_temperature"])

        self.network_arch = args["network_arch"]

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.mask_type = args["mask_type"]  # "weight mask"
        self.conv_mask_type = args["conv_mask_type"]

        self.use_sigma_k = args["use_sigma_k"]

        self.use_weighted_kld_ber = args.get("use_weighted_kld_ber", False)

        # additionally multiplied by kl_weight - so setting it to 1e-6 here will down_weight kld_bernoulli by 1e-7
        # when kl_weight is 1e-1
        self.kld_ber_multiplier = args.get("kld_ber_multiplier", 1)

        # accumulates sample based kl divergence between concrete Bernoullis
        self.kld_bernoulli = 0

        self.eps = 0.01
        self.threshold = 0.05
        self.weight_shape_ls = weight_shape_ls

        self.build_sampler()

    def build_sampler(self):
        # inverse softplus to avoid parameter underflow
        alpha = np.log(np.exp(self.args["a_prior"]) - 1)
        beta = np.log(np.exp(self.args["b_prior"]) - 1)

        self.conv_weight_shape_ls, self.fc_weight_shape_ls = self.weight_shape_ls
        self.n_conv_layers = len(self.conv_weight_shape_ls)
        self.n_fc_layers = len(self.fc_weight_shape_ls)
        self.n_layers = self.n_conv_layers + self.n_fc_layers

        # Hyper-parameters for prior beta
        self.alpha = nn.Parameter(torch.full((self.n_layers,), alpha), requires_grad=False)
        self.beta = nn.Parameter(torch.full((self.n_layers,), beta), requires_grad=False)

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.full((self.n_layers,), alpha))
        self.b_k = nn.Parameter(torch.full((self.n_layers,), beta))

        if self.use_sigma_k:
            sigma_k_ls = []
            self.sigma_k_init = 0.1

            for shape in self.conv_weight_shape_ls + self.fc_weight_shape_ls:
                shape = shape[:2] if self.conv_mask_type == "single_per_channel" else shape

                sigma_k_ls.append(
                    nn.Parameter(torch.logit(torch.ones(shape) * self.sigma_k_init), requires_grad=True))
            self.sigma_k = nn.ParameterList(sigma_k_ls)

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

    @property
    def expand_pi_(self):
        # x: shape (n_samples, K)
        weight_shape_ls = self.conv_weight_shape_ls + self.fc_weight_shape_ls
        weight_shape_ls = [shape[:2] if self.conv_mask_type == "single_per_channel" else shape for shape in
                           weight_shape_ls]
        return lambda \
                x: [x[:, l][[..., ] + [None, ] * len(shape)].expand(-1, *shape) for l, shape in
                    enumerate(weight_shape_ls)]

    def forward(self, n_samples=None, get_pi=False, get_single_mask=False):
        n_samples = (
            self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples

        # sample from variational beta distribution
        ν = self.variational_beta.rsample((n_samples,)).view(n_samples, self.n_layers)  # n_samples x L
        pi = torch.cumsum(ν.log(), dim=1).exp()  # n_samples x L

        pi_expanded = self.pi_expander(pi)  # n_samples x L x M x M

        if self.use_sigma_k:
            pi_expanded = self.add_sigma_k_pi_expanded(pi_expanded)

        Z = self.sample_mask(pi_expanded, get_single_mask, pi)

        # Z shape: L x n_samples x M x N
        threshold_Z = torch.vstack([(Z_l > self.eps).sum(-1).sum(-1) for Z_l in Z]).T
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        threshold = max(threshold_array)

        if get_pi:
            keep_prob = pi.detach().mean(0)
            return Z, threshold, keep_prob

        return Z, threshold

    def add_sigma_k_pi_expanded(self, pi_expanded):
        pi_expanded_ = []
        for pi_expanded_l, sigma_k_l in zip(pi_expanded, self.sigma_k):
            logit_l = torch.logit(pi_expanded_l)
            logit_l_ = logit_l + sigma_k_l.unsqueeze(0)
            pi_first_layer = torch.sigmoid(logit_l_)
            pi_expanded_.append(pi_first_layer)
        return pi_expanded_

    def sample_mask(self, pi_expanded, get_single_mask, pi):
        Z = self.sample_weight_mask(pi_expanded, get_single_mask, self.pi_expander(pi))
        return Z

    def sample_weight_mask(self, pi, get_single_mask=False, pi_prior=None):
        Z_ls = []
        if self.training:
            self.kld_bernoulli = 0
            for l, pi_l in enumerate(pi):
                con_ber_l = RelaxedBernoulli(probs=pi_l, temperature=self.posterior_temperature)
                Z_l = con_ber_l.rsample()
                Z_ls.append(Z_l)
                if self.use_sigma_k:
                    if self.use_weighted_kld_ber:
                        self.kld_bernoulli += self.calculate_kl_divergence_bernoulli(
                            pi_l, pi_prior[l], Z_l) * self.kld_ber_multiplier
                    else:
                        self.kld_bernoulli = self.calculate_kl_divergence_bernoulli(pi_l, pi_prior[l], Z_l)

        elif get_single_mask:
            # calculates mean of the pis and generates single mask
            Z_ls = []
            for l, pi_l in enumerate(pi):
                pi_l = pi_l.mean(0)
                con_ber_l = Bernoulli(probs=pi_l)
                Z_l = con_ber_l.sample().unsqueeze(dim=0)
                Z_ls.append(Z_l)
        else:
            Z_ls = []
            for l, pi_l in enumerate(pi):
                con_ber_l = Bernoulli(probs=pi_l)
                Z_l = con_ber_l.sample()
                Z_ls.append(Z_l)
        return Z_ls

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

        kld = (log_prob_posterior - log_prob_prior)  # S x K x M x M
        return kld.mean(0).sum()

    def reset_for_new_task(self, model_usage=None):
        if model_usage is None:
            self.alpha.data.copy_(self.a_k.data)
            self.beta.data.copy_(self.b_k.data)

        else:
            number_activated_layers = (model_usage > 0.025).sum()
            model_usage = np.clip(model_usage.cpu().numpy(), 1e-5, 1)
            pseudo_vs = torch.tensor(
                np.clip(model_usage[1:] / model_usage[:-1], 1e-3, 1 - 1e-3))[:number_activated_layers - 1].to(
                self.alpha.device)

            pseudo_alphas = F.softplus(self.a_k.data[1:number_activated_layers]) + self.eps
            pseudo_betas = pseudo_alphas * (1 - pseudo_vs) / pseudo_vs

            self.alpha.data.copy_(self.a_k.data)
            self.beta.data.copy_(self.b_k.data)
            self.beta.data[1:number_activated_layers].copy_(pseudo_betas)

        if self.use_sigma_k:
            for sigma_k in self.sigma_k:
                sigma_k.data.copy_(torch.logit(torch.ones_like(sigma_k) * self.sigma_k_init))

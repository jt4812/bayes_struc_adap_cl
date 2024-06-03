import torch
from torch import nn
from torch.distributions import kl_divergence, Normal

from .decoder_adaptive import BayesAdaptiveDecoder
from .encoder import Encoder


class BayesAdaptiveVAE(nn.Module):
    def __init__(self, args, exp_dict=None):
        super(BayesAdaptiveVAE, self).__init__()
        self.model_args = args
        self.exp_dict = exp_dict

        self.encoder = nn.ModuleList([Encoder(args, exp_dict) for _ in range(exp_dict["n_tasks"])])
        self.decoder = BayesAdaptiveDecoder(args, exp_dict)

        self.latent_feature_dim = args["latent_feature_dim"]

        self.n_train_samples = args["n_train_samples"]
        self.n_test_samples = args["n_test_samples"]

        self.kl_weight = args["kl_weight"]
        self.n_tasks = exp_dict["n_tasks"]

        self.device = args["device"]
        self.prior_dist = Normal(torch.zeros(self.latent_feature_dim).to(self.device),
                                 torch.ones(self.latent_feature_dim).to(self.device))

    def forward(self, x, task_idx, n_samples=None, train=True, sample_W=True):
        n_samples = (self.n_train_samples if self.training else self.n_test_samples) if n_samples is None else n_samples
        mu, log_sig = self.encode(x, task_idx)
        z_ls, KLD_z = self.reparameterize(mu, log_sig, n_samples)

        x_recons_ls = []
        for s in range(n_samples):
            x_recons = self.decode(z_ls[s], task_idx, sample_W=sample_W)
            x_recons_ls.append(x_recons)
        return torch.concatenate(x_recons_ls), KLD_z

    def encode(self, x, task_idx):
        mu, log_var = self.encoder[task_idx](x)
        return mu, log_var

    def decode(self, z, task_idx, sample_W=True):
        x_ = self.decoder(z, task_idx, sample_W=sample_W)
        return x_

    def reparameterize(self, mu, log_sig, n_samples=1):
        variational_dist_ = Normal(mu, log_sig.exp())

        mu = mu.unsqueeze(0).expand((n_samples, -1, -1))
        log_sig = log_sig.unsqueeze(0).expand((n_samples, -1, -1))

        variational_dist = Normal(mu, log_sig.exp())
        z_ls = variational_dist.rsample()

        # sum dim 1 to calculate kl divergence for each z sample and mean dim 0 to average across data points
        KLD = kl_divergence(variational_dist_, self.prior_dist).sum(dim=1).mean(0)
        return z_ls, KLD

    def generate_samples(self, task_idx, n_samples=100):
        x_samples = self.decoder.generate_samples(task_idx, n_samples)
        return x_samples

    def get_E_loglike(self, neg_loglike_fun, x, x_recons):
        n_samples = len(x_recons) // len(x)
        batch_sze = len(x)
        x = x.squeeze().repeat(n_samples, 1)
        x_recons = x_recons.view(n_samples * batch_sze, -1)

        # sum along dim 1 to calculate recons_error for each pixel
        neg_loglike = neg_loglike_fun(x_recons, x).sum(dim=1)

        # average for n * samples * batchsize to calculate per datapoint per sample loss
        E_neg_loglike = neg_loglike.mean(dim=0)
        return E_neg_loglike

    def estimate_total_loss(self, neg_loglike_fun, x, x_recons, KLD_z, kl_scale, task_id):
        # x_recons shape: n_samples x batch_size x n_classes
        E_neg_loglike = self.get_E_loglike(neg_loglike_fun, x, x_recons)
        kld_weight = self.decoder.kl_divergence_weight(task_id)  # weight KLD
        kld_struc = self.decoder.kl_divergence_struc()  # struc KLD
        total_loss = E_neg_loglike + KLD_z + (kld_weight + kld_struc) * kl_scale
        return total_loss, E_neg_loglike, KLD_z, kld_weight * kl_scale, kld_struc * kl_scale

    def reset_for_new_task(self):
        # for encoder_task in self.encoder:
        #     encoder_task.reset_for_new_task()
        self.decoder.reset_for_new_task()

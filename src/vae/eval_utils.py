import torch

from src.vae.models.vae_adaptive import BayesAdaptiveVAE


def get_loglike(neg_loglike_fun, x, x_recons):
    n_samples = len(x_recons) // len(x)
    batch_size = len(x)
    x = x.squeeze().repeat(n_samples, 1)
    x_recons = x_recons.view(n_samples * batch_size, -1)

    # sum along dim 1 to calculate recons_error for each pixel
    neg_loglike = neg_loglike_fun(x_recons, x).sum(dim=1)

    # view in n_samples x batch_size shape
    return neg_loglike.view(n_samples, batch_size)


def estimate_ll(neg_loglike_fun, x, x_recons, KLD_z):
    # x_recons shape: n_samples x batch_size
    loglike = - get_loglike(neg_loglike_fun, x, x_recons)

    bound = loglike.clone() - KLD_z

    bound_max = torch.max(bound, dim=0)[0]
    bound -= bound_max
    log_norm = torch.log(torch.clip(torch.mean(torch.exp(bound), 0), 1e-9, torch.inf))
    test_ll = log_norm + bound_max
    mean_test_ll = test_ll.mean()

    return mean_test_ll


def calculate_lower_bound(model: BayesAdaptiveVAE, data_loader, task_idx, loss_fn, n_samples=5000):
    lower_bound = 0.0
    model.eval()
    for i, (data, *_) in enumerate(data_loader):
        data = data.float().to(model.device)
        with torch.no_grad():
            data_recons, KLD_z = model(data.clone(), task_idx=task_idx, n_samples=n_samples, sample_W=False)
            test_ll = estimate_ll(loss_fn, data, data_recons, KLD_z)

        lower_bound += - test_ll.item() / len(data_loader)
    return lower_bound, KLD_z

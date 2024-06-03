def train_single_epoch_bayes_adaptive(model, train_loader, optimizer, loss_fn, task_idx, is_hp_search=False):
    model.train()

    train_loss = 0.0
    neg_loglike = 0.0
    latent_kld = 0.0
    weight_kld = 0.0
    kld_struc = 0.0

    for i, data_ in enumerate(train_loader):
        data = data_[0].float().to(model.device)

        optimizer.zero_grad()
        data_recons, KLD_z = model(data.clone(), task_idx=task_idx)
        loss, E_neg_loglike, KLD_z, weight_kld_, kld_struc_ = model.estimate_total_loss(
            loss_fn, data, data_recons, KLD_z,
            kl_scale=1 / len(train_loader.dataset),
            task_id=task_idx
        )
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        neg_loglike += E_neg_loglike.item() / len(train_loader)
        latent_kld += KLD_z.item() / len(train_loader)
        weight_kld += weight_kld_.item() / len(train_loader)
        kld_struc += kld_struc_.item() / len(train_loader)

    return train_loss, neg_loglike, latent_kld, weight_kld, kld_struc


def generate_samples(model, task_idx, n_samples):
    sample_gens = model.generate_samples(task_idx, n_samples)
    return sample_gens


def generate_samples_until_now(model, task_idx, n_samples):
    gen_imgs_ls = []
    for task_idx_ in range(task_idx + 1):
        sample_gens = model.generate_samples(task_idx_, n_samples)
        gen_imgs_ls.append(sample_gens)
    return gen_imgs_ls

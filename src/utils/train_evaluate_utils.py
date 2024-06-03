import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_single_epoch(model, train_loader, optimizer, loss_fn, task_idx=None, use_fixed_mask=False,
                       is_hp_search=True, ):
    train_loss = 0.0
    model.train()
    tq = tqdm(enumerate(train_loader)) if not is_hp_search else enumerate(train_loader)

    for iter_idx, data_ in tq:
        data, labels, task_ids = data_
        data = data.float().to(model.device)
        if len(labels.shape) == 1:  # no one-hot encoding for class
            labels = labels.long().to(model.device)
        elif len(labels.shape) == 2:  # one hot encoding for class
            labels = labels.float().to(model.device)

        optimizer.zero_grad()

        act_vec, threshold_ = model(data, ret_threshold=True, task_idx=task_idx, use_fixed_mask=use_fixed_mask)

        if len(labels.shape) == 2:
            labels = torch.argmax(labels, 1)

        loss, E_neg_loglike, weight_kld, struct_kld = model.estimate_total_loss(
            loss_fn, act_vec, labels,
            kl_scale=1 / len(train_loader.dataset),
            use_fixed_mask=use_fixed_mask,
            task_idx=task_idx
        )

        loss.backward()
        optimizer.step()

        # adding losses
        train_loss += loss.item()
        if not is_hp_search:
            tq.set_postfix(
                {"train_loss": loss.item(), "neg_loglike": E_neg_loglike.item()})

    return train_loss


def train_single_epoch_vae(model, train_loader, optimizer, loss_fn, task_idx=None, use_fixed_mask=False):
    train_loss = 0.0

    model.train()
    with tqdm(enumerate(train_loader)) as tq:
        for i, data_ in tq:
            data = data_[0].float().to(model.device)
            task_ids = None

            optimizer.zero_grad()
            data_recons, threshold, KLD_z = model(
                data, ret_threshold=True, task_idx=task_idx, use_fixed_mask=use_fixed_mask)
            loss, E_neg_loglike, weight_kld, struct_kld = model.estimate_ll(
                loss_fn, data, data_recons, KLD_z,
                task_ids=task_ids,
                kl_scale=1 / len(train_loader.dataset),
                use_fixed_mask=use_fixed_mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            tq.set_postfix(
                {"train_loss": loss.item(), "neg_loglike": E_neg_loglike.item()})
    return train_loss


def evaluate(model, test_loader, task_idx=None, use_fixed_mask=False):
    correct_pred_count = 0

    with torch.no_grad():
        model.eval()
        for i, data_ in enumerate(test_loader):
            if len(data_) == 2:
                data, labels = data_
            else:
                data, labels, _ = data_

            data = data.float().to(model.device)
            labels = labels.long().to(model.device)
            pred = model(data, task_idx=task_idx, use_fixed_mask=use_fixed_mask)
            logits = pred.mean(0)
            if len(logits.squeeze().shape) == 3:
                logits = logits[task_idx]  # use only the output corresponding to the task_idx

            assert len(logits.shape) == 2
            probs = F.softmax(logits, dim=1)

            predicted = torch.argmax(probs, 1)

            if len(labels.shape) == 2:
                labels = torch.argmax(labels, 1)

            correct_pred_count += (predicted == labels).sum().item()

    avg_acc = correct_pred_count / len(test_loader.dataset)

    test_metrics = {'test_acc': round(avg_acc * 100, 3), }
    return test_metrics

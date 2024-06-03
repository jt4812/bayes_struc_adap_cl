from torch import optim

from src.models.struc_adap_alexnet import BayesAdaptiveCNN
from src.models.struc_adap_alexnet_long import BayesAdaptiveCNN as BayesAdaptiveCNNLong
from src.models.struc_adap_model import BayesAdaptiveMLP
from src.vae.models.vae_adaptive import BayesAdaptiveVAE


def get_optimizer_(model, lr_weight=0.01, lr_struc=0.1, optimizer_type="adam", momentum=0.10, decay_rate_weight=1,
                   decay_rate_struc=1, task_idx=0, **kwargs):
    lr_weight = lr_weight * (decay_rate_weight ** task_idx)
    lr_struc = lr_struc * (decay_rate_struc ** task_idx)
    weight_decay = 0

    if isinstance(model, BayesAdaptiveVAE):
        params = [
            {"params": model.decoder.structure_sampler.parameters(), "lr": lr_struc},
            {"params": model.decoder.heads.parameters(), "lr": lr_weight},
            {"params": model.decoder.shared_decoder.parameters(), "lr": lr_weight},
            {"params": model.decoder.final_layer_decoder.parameters(), "lr": lr_weight},
            {"params": model.encoder.parameters(), "lr": lr_weight},
        ]

    elif isinstance(model, BayesAdaptiveCNN) or isinstance(model, BayesAdaptiveCNNLong):
        params = [
            {"params": model.conv_layers.parameters(), "lr": lr_weight},
            {"params": model.out_layer.parameters(), "lr": lr_weight},
        ]
        if not isinstance(model, BayesAdaptiveCNNLong) and hasattr(model, "fc_layers") and len(model.fc_layers) > 0:
            params.append({"params": model.fc_layers.parameters(), "lr": lr_weight}, )

        if hasattr(model, "structure_sampler"):
            params.append({"params": model.structure_sampler.parameters(), "lr": lr_struc}, )

    elif isinstance(model, BayesAdaptiveMLP):
        params = [
            {"params": model.layers.parameters(), "lr": lr_weight},
            {"params": model.out_layer.parameters(), "lr": lr_weight},
        ]
        if hasattr(model, "structure_sampler"):
            params.append({"params": model.structure_sampler.parameters(), "lr": lr_struc}, )
    else:
        raise NotImplemented()

    if optimizer_type == "adam":
        return optim.Adam(params, lr_weight, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        params = [
            {"params": model.structure_sampler.parameters(), "lr": lr_struc, "momentum": momentum},
            {"params": model.layers.parameters(), "lr": lr_weight, "momentum": momentum},
            {"params": model.out_layer.parameters(), "lr": lr_weight, "momentum": momentum},
        ]
        return optim.SGD(params, lr_weight)
    else:
        raise Exception(f"{optimizer_type} not defined.")


def get_optimizer(model, optimizer_args, t_idx=0):
    if isinstance(optimizer_args, optim.Optimizer):
        optimizer = optimizer_args
    else:
        optimizer = get_optimizer_(model, task_idx=t_idx, **optimizer_args)
    return optimizer

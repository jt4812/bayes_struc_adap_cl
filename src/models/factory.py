from src.models.struc_adap_alexnet import BayesAdaptiveCNN
from src.models.struc_adap_alexnet_long import BayesAdaptiveCNN as BayesAdaptiveCNNLong
from src.models.struc_adap_model import BayesAdaptiveMLP
from src.vae.models.vae_adaptive import BayesAdaptiveVAE


def get_model(exp_dict, model_args):
    if exp_dict["experiment"] == "vae_adaptive_vcl":
        model = BayesAdaptiveVAE(model_args, exp_dict)
    elif model_args["network_arch"] == "fc":
        model = BayesAdaptiveMLP(model_args, exp_dict)
    elif model_args["network_arch"] == "alexnet":
        model = BayesAdaptiveCNN(model_args, exp_dict)
    elif model_args["network_arch"] == "alexnet_long":
        model = BayesAdaptiveCNNLong(model_args, exp_dict)
    else:
        raise NotImplementedError()
    return model.to(model_args["device"])

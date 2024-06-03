from torch import nn


class Encoder(nn.Module):
    def __init__(self, args, exp_dict=None):
        super(Encoder, self).__init__()

        self.input_feature_dim = args["input_feature_dim"]
        self.latent_feature_dim = args["latent_feature_dim"]
        self.max_width = args["max_width"]

        self.n_layers = args["n_head_layers"] * 2 - 1

        self.backbone = self.create_network()

        self.mu_layer = nn.Linear(self.max_width, self.latent_feature_dim)
        self.log_var_layer = nn.Linear(self.max_width, self.latent_feature_dim)

    def forward(self, x):
        x = self.backbone(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var

    def create_network(self):
        input_feature_dim, max_width, n_layers = self.input_feature_dim, self.max_width, self.n_layers
        model_size = [input_feature_dim, ] + [max_width] * n_layers
        shared_decoder_layers = []
        for layer_idx in range(n_layers):
            prev_layer_width = model_size[layer_idx]
            next_layer_width = model_size[layer_idx + 1]

            layer_ = nn.Linear(prev_layer_width, next_layer_width)
            shared_decoder_layers.append(layer_)
            shared_decoder_layers.append(nn.ReLU())

        return nn.Sequential(*shared_decoder_layers)

    def initialize(self):
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.xavier_uniform_(self.log_var_layer.weight)

    def reset_for_new_task(self):
        self.initialize()

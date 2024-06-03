import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.prev_weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.prev_bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

        self.eps = 1e-8
        nn.init.uniform_(self.weight, -0.2, 0.2)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, mask=None):
        weight = self.weight
        bias = self.bias
        if mask is not None:
            weight = self.weight * mask
            bias = self.bias * mask.max(dim=1)[0]

        return F.linear(x, weight, bias)

    @property
    def kl_divergence(self):
        return ((self.weight - self.prev_weight) ** 2).sum() + ((self.bias - self.prev_bias) ** 2).sum()

    def masked_kl_divergence(self, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1).unsqueeze(-1)

        prev_weight = self.prev_weight * mask
        weight = self.weight * mask

        return ((weight - prev_weight) ** 2).sum() + ((self.bias - self.prev_bias) ** 2).sum()

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
        if mask is None:
            self.prev_weight.data.copy_(self.weight.data)
            self.prev_bias.data.copy_(self.bias.data)

        elif mask_type == "weight_mask":
            self.prev_weight.data.copy_(mask * self.weight.data + (1 - mask) * self.prev_weight.data)
            self.prev_bias.data.copy_(self.bias.data)

    def get_weight_shape(self):
        return self.out_features, self.in_features


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, has_norm=False, residual=False,
                 act_layer_fn=None):
        super(MLPBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features)
        if act_layer_fn == "sigmoid":
            self.act_layer_fn = nn.Sigmoid()
        elif act_layer_fn == "relu":
            self.act_layer_fn = nn.ReLU(inplace=False)
        else:
            self.act_layer_fn = nn.LeakyReLU(inplace=False)

        self.has_norm = has_norm
        if has_norm:
            self.norm_layer = nn.BatchNorm1d(out_features)
        self.residual = residual

    def forward(self, x, mask=None):
        residual = x

        if mask is not None:
            assert len(mask.shape) in [1, 2]
            use_neuron_mask = (len(mask.shape) == 1)

        if (mask is not None) and use_neuron_mask:
            out = self.linear(x)
        else:
            out = self.linear(x, mask=mask)

        if self.has_norm:
            out = self.norm_layer(out)

        out = self.act_layer_fn(out)

        if (mask is not None) and use_neuron_mask:
            out *= mask.view(1, -1)

        if self.residual:
            out = out + residual

        return out

    # @property
    # def kl_divergence(self):
    #     return self.linear.kl_divergence
    #
    # def masked_kl_divergence(self, mask):
    #     return self.linear.masked_kl_divergence(mask)
    #
    # def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
    #     return self.linear.reset_for_new_task(mask, mask_type)
    #

    @property
    def kl_divergence(self):
        pass

    def masked_kl_divergence(self, mask):
        pass

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask"):
        pass

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, has_norm={}, has_residual={}'.format(
            self.in_features, self.out_features, self.has_norm, self.residual)

    def get_weight_shape(self):
        return self.linear.get_weight_shape()

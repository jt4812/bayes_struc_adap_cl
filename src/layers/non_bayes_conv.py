import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.prev_weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel), requires_grad=False)
        self.prev_bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)

        self.eps = 1e-8
        nn.init.uniform_(self.weight, -0.2, 0.2)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, mask=None, task_idx=None):
        weight = self.weight
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            weight = weight * mask
        return F.conv2d(x, weight, self.bias, padding=self.padding, stride=self.stride)

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

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, kernel={}, stride={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride)

    def get_weight_shape(self):
        return (self.out_channels, self.in_channels, *self.kernel)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, has_norm=False,
                 has_max_pool=True, residual=False):
        super(ConvBlock, self).__init__()

        self.conv = Conv(in_channels, out_channels, kernel, (1, 1) if has_max_pool else stride, padding)
        self.act_layer_fn = nn.LeakyReLU()

        self.has_norm = has_norm
        if has_norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)

        self.has_max_pool = has_max_pool
        if has_max_pool:
            self.max_layer = nn.MaxPool2d(stride)
        self.residual = residual

    def forward(self, x, mask=None, task_idx=None):
        residual = x

        out = self.conv(x, mask=mask)

        if self.has_norm:
            out = self.norm_layer(out)

        out = self.act_layer_fn(out)

        if self.has_max_pool:
            out = self.max_layer(out)

        if self.residual:
            out += residual

        return out

    def get_weight_shape(self):
        return self.conv.get_weight_shape()

    # @property
    # def kl_divergence(self):
    #     return self.conv.kl_divergence
    #
    # def masked_kl_divergence(self, mask):
    #     return self.conv.masked_kl_divergence(mask)
    #
    # def reset_for_new_task(self, mask=None, mask_type="neuron_mask", task_idx=None):
    #     self.conv.reset_for_new_task(mask, mask_type)
    @property
    def kl_divergence(self):
        pass

    def masked_kl_divergence(self, mask):
        pass

    def reset_for_new_task(self, mask=None, mask_type="neuron_mask", task_idx=None):
        pass

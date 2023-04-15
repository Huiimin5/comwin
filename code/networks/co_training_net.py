import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from .vnet import VNet

import torch.distributions as td
import torch

class TriVNet(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False):
        super().__init__()
        self.branch1 = VNet(input_channels,num_classes,n_filters,normalization, has_dropout)
        self.branch2 = VNet(input_channels,num_classes,n_filters,normalization, has_dropout)
        self.branch3 = VNet(input_channels,num_classes,n_filters,normalization, has_dropout)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
        elif step == 3:
            return self.branch3(data)


from .vnet import VNet_dsba_after8
class TriDSBAVNet_after8(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='batchnorm',
                 head_type=1,
                 window_size = 2,
                 self_atten_head_num=1,
                 sparse_attn = False,
                 dilated_windows=False,
                 has_dropout=False):
        super().__init__()
        self.branch1 = VNet_dsba_after8(input_channels,num_classes,n_filters,normalization, head_type, window_size, self_atten_head_num, sparse_attn, dilated_windows, has_dropout)
        self.branch2 = VNet_dsba_after8(input_channels,num_classes,n_filters,normalization, head_type, window_size, self_atten_head_num, sparse_attn, dilated_windows, has_dropout)
        self.branch3 = VNet_dsba_after8(input_channels,num_classes,n_filters,normalization, head_type, window_size, self_atten_head_num, sparse_attn, dilated_windows, has_dropout)

    def forward(self, data, pseudo_labels=None, step=1, foward_step=1):
        # if not self.training:
        #     pred1 = self.branch1(data)
        #     return pred1
        if not self.training:

            x1_test_1, x8_test_1, out_at8_test_1 = self.branch1(data, step=1)
            _, max_test_1 = torch.max(out_at8_test_1, dim=1)
            x8_after_test_1 = self.branch1(x8_test_1, max_test_1, step=2)
            logits_test_1 = self.branch1(x1_test_1, x8_after_test_1, step=3)
            return logits_test_1
        if foward_step == 1:
            if step == 1:
                return self.branch1(data, step=foward_step)
            elif step == 2:
                return self.branch2(data, step=foward_step)
            elif step == 3:
                return self.branch3(data, step=foward_step)
        elif foward_step == 2:
            if step == 1:
                return self.branch1(data, pseudo_labels, step=foward_step)
            elif step == 2:
                return self.branch2(data, pseudo_labels, step=foward_step)
            elif step == 3:
                return self.branch3(data, pseudo_labels, step=foward_step)
        else:
            assert foward_step == 3
            if step == 1:
                return self.branch1(*data, step=foward_step)
            elif step == 2:
                return self.branch2(*data, step=foward_step)
            elif step == 3:
                return self.branch3(*data, step=foward_step)

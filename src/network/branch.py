from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn
from MOC_utils.Temporal_Shift import TemporalShift, LightweightTemporalAttentionModule
# import paddle.nn.functional as F


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

        self.K =K

        self.shift = TemporalShift(n_segment=K, n_div=8)
        self.attention = LightweightTemporalAttentionModule(num_channels=input_channel, reduction_ratio=4)

    def forward(self, input_chunk):
        output = {}
        output_wh = []
        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        # # 原论文直接通道维度进行拼接后，进行卷积操作
        # input_chunk = torch.cat(input_chunk, dim=1)
        # output_wh = torch.cat(output_wh, dim=1)

        # 修改后，增加了两个模块，轻量级注意力模块和temporal_shift模块
        input_chunk = torch.stack(input_chunk, dim=1)
        # 使用轻量级注意力模块
        input_chunk = self.attention(input_chunk)
        input_chunk = input_chunk.view(-1, input_chunk.size(2), input_chunk.size(3), input_chunk.size(4))
        # 使用temporal_shift模块，进行时间维度的处理
        input_chunk = self.shift(input_chunk)
        input_chunk = torch.split(input_chunk, int((input_chunk.size(0))/self.K), dim=0)
        input_chunk = list(input_chunk)
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)

        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['wh'] = output_wh
        return output

'''
Blog:
'''

import torch
import torch.nn as nn


class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class DAAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DAAttention, self).__init__()

        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.pam = PositionAttentionModule(out_planes)
        self.cam = ChannelAttentionModule()
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c
        return feat_fusion


if __name__ == '__main__':
    feature = torch.randn(12, 512, 30, 30)
    a2attention = DAAttention(in_planes=512, out_planes=128)
    out = a2attention(feature)
    print(out.shape)

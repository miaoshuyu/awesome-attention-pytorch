'''
Blog:https://www.cnblogs.com/bonelee/p/9030092.html
'''

import torch
import torch.nn as nn


class SElayer(nn.Module):
    def __init__(self, channel, rediction=16):
        super(SElayer, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // rediction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // rediction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.globalavgpool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        return x * y


if __name__ == '__main__':
    feature = torch.randn(12, 512, 30, 30)
    selayer = SElayer(channel=512)
    out = selayer(feature)
    print(out.shape)

'''
Blog:https://blog.csdn.net/qq_37014750/article/details/83989334
'''

import torch
import torch.nn as nn


class A2Attention(nn.Module):
    def __init__(self, in_planes):
        super(A2Attention, self).__init__()
        self.dimension_reduction = nn.Conv2d(in_channels=in_planes, out_channels=in_planes // 2, kernel_size=1,
                                             stride=1)
        self.Conv1x1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, groups=in_planes)
        self.softmax = nn.Softmax(dim=1)
        self.dimension_extension = nn.Conv2d(in_channels=in_planes // 2, out_channels=in_planes, kernel_size=1,
                                             stride=1)

    def forward(self, x):
        temp = x
        # gather
        # two branch convolution
        gather1 = self.dimension_reduction(temp)
        gather1 = gather1.view(gather1.size(0), gather1.size(1), -1)

        gather2 = self.softmax(self.Conv1x1(temp))
        gather2 = gather2.view(gather2.size(0), gather2.size(1), -1)
        gather2 = torch.transpose(gather2, 1, 2)

        # bilinear pooling
        gather = torch.bmm(gather1, gather2) / gather1.size(2)  # torch.Size([13, 32, 64])
        gather = gather.view(gather.size(0), -1)
        gather = torch.sign(gather) * torch.sqrt(torch.abs(gather) + 1e-5)
        # gather = torch.sqrt(gather + 1e-5)
        gather = torch.nn.functional.normalize(gather)
        gather = gather.view(gather.size(0), gather1.size(1), gather2.size(2))

        #  distribution
        distribution2 = self.softmax(self.Conv1x1(temp))
        distribution2 = distribution2.view(distribution2.size(0), distribution2.size(1), -1)

        # feature distribution
        out = torch.bmm(gather, distribution2)
        out = out.view(out.size(0), out.size(1), x.size(2), x.size(3))
        out = self.dimension_extension(out)
        out = out + x
        return out


if __name__ == '__main__':
    feature = torch.randn(12, 512, 30, 30)
    a2attention = A2Attention(in_planes=512)
    out = a2attention(feature)
    print(out.shape)

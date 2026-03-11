import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish activation.
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class CMF_Block(nn.Module):
    def __init__(self, in_channel, groups=4):
        super(CMF_Block, self).__init__()

        # Generate query, key, and value with grouped convolutions.
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, groups=groups)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, groups=groups)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, groups=groups)

        self.scale = in_channel ** -0.5

        # Add an MLP branch for attention-map refinement.
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            Mish(),
            nn.Linear(in_channel // 2, in_channel)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
            Mish()
        )

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape == ir_fea.shape, "Input feature maps must have the same shape."
        bs, c, h, w = rgb_fea.shape

        q = self.conv1(rgb_fea)
        k = self.conv2(ir_fea)
        v = self.conv3(ir_fea)

        q = q.view(bs, c, h * w).transpose(-2, -1)
        k = k.view(bs, c, h * w)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Optionally refine the attention matrix with the MLP.
        # attn = attn.transpose(-2, -1).reshape(bs * h * w, c)
        # attn = self.mlp(attn).reshape(bs, h * w, h * w).transpose(-2, -1)

        v = v.view(bs, c, h * w).transpose(-2, -1)
        z = torch.matmul(attn, v)

        z = z.transpose(-2, -1).view(bs, c, h, w)
        output = rgb_fea + self.conv4(z)

        return output
    

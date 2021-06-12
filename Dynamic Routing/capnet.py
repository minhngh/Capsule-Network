import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, dim = -1):
    square_norm = torch.sum(x ** 2, dim = -1, keepdims = True)
    return square_norm / (1 + square_norm) * x / (torch.sqrt(square_norm) + 1e-6)
    

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, cap_dim, kernel_size = 9, stride = 2, padding = 0):
        super().__init__()
        self.cap_dim = cap_dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
    def forward(self, x):
        assert x.shape[1] % self.cap_dim == 0
        x = self.conv(x)
        number_capsules = x.shape[1] // self.cap_dim
        x = x.view(-1, number_capsules, *(x.shape[2:4]), self.cap_dim)
        return squash(x)
class DigitCapsule(nn.Module):
    def __init__(self, in_capsules_shape, in_cap_dim, out_capsules, out_cap_dim, iterations = 3):
        super().__init__()
        self.in_capsules_shape = in_capsules_shape
        self.in_cap_dim = in_cap_dim
        self.out_capsules = out_capsules
        self.out_cap_dim = out_cap_dim
        self.iterations = iterations
        self.W = nn.Parameter(torch.randn(1, *in_capsules_shape, out_capsules, in_cap_dim, out_cap_dim))
    def forward(self, x):
        # shape of x: N x 32 x 6 x 6 x 1  x  1 x 8
        # shapw of W: 1 x 32 x 6 x 6 x 10 x 8 x 16
        # -> N x 32 x 6 x 6 x 10 x 1 x 16
        # squeeze -> N x 32 x 6 x 6 x 10 x 16
        u_hat = torch.matmul(x.unsqueeze(-2).unsqueeze(-2), self.W).squeeze(-2)
       
        b = torch.zeros(x.shape[0], *self.in_capsules_shape, self.out_capsules)
        for _ in range(self.iterations - 1):
            # shape of c: N x 32 x 6 x 6 x 10
            c = F.softmax(b, dim = -1)
            # c:     N x 32 x 6 x 6 x 10 x 1
            # element-wise
            # u_hat: N x 32 x 6 x 6 x 10 x 16
            # -> N x 32 x 6 x 6 x 10 x 16
            # sum -> N x 10 x 16 
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim = [1, 2, 3])
            # v: N x 10 x 16
            v = squash(s)
            # N x 32 x 6 x 6 x 10 x 16
            # element-wise
            # N x 1  x 1 x 1 x 10 x 16
            # -> N x 32 x 6 x 6 x 10 x 16
            # sum -> N x 32 x 6 x 6 x 10
            b += torch.sum(u_hat * v.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim = -1)

        c = F.softmax(b, dim = -1)
        s = torch.sum(c.unsqueeze(-1) * u_hat, dim = [1, 2, 3])
        v = squash(s)
        return v


class CapNet(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size = 9, stride = 1)
        self.relu1 = nn.ReLU()
        self.primary_cap = PrimaryCapsule(256, 256, 8)
        self.digit_cap = DigitCapsule((32, 6, 6), 8, 10, 16)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.primary_cap(x)
        x = self.digit_cap(x)
        return x

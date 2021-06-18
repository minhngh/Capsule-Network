import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, dim = -1):
    square_norm = torch.sum(x ** 2, dim = -1, keepdims = True)
    return square_norm / (1 + square_norm) * x / (torch.sqrt(square_norm) + 1e-6)

def init_c(kernel_map, parent_shape, in_capsules, out_capsules):
    num_parents_per_child = kernel_map.sum(axis = 0, keepdims = True)
    initial_c = kernel_map / (num_parents_per_child * out_capsules + 1e-6)
    initial_c = initial_c[kernel_map == 1].reshape(*parent_shape, -1)[..., None, None]
    initial_c = np.tile(initial_c, [1, 1, 1, in_capsules, out_capsules])[None]
    return initial_c
def update_c(b):
    #N, ps, ps, ks * ks, in_caps, out_caps
    bs = b.size(0)
    ph = b.size(1)
    pw = b.size(2)
    ks2 = b.size(3)
    in_caps = b.size(-2)
    b = b.permute(0, 3, 4, 1, 2, 5).view(bs, ks2, in_caps, -1)
    c = F.softmax(b, dim = -1)
    c = c.view(bs, ks2, in_caps, ph, pw, -1)
    c = c.permute(0, 3, 4, 1, 2, 5)
    return c

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, cap_dim, kernel_size = 9, stride = 2, padding = 0):
        super().__init__()
        self.cap_dim = cap_dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
    def forward(self, x):
        assert x.shape[1] % self.cap_dim == 0
        x = self.conv(x)
        number_capsules = x.shape[1] // self.cap_dim
        x = x.view(-1, *(x.shape[2:4]), number_capsules, self.cap_dim)
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
        for i in range(self.iterations - 1):
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


# class ConvCapsule(nn.Module):
#     def __init__(self, input_shape, in_capsules, in_cap_dim, out_capsules, out_cap_dim, kernel_size, stride, padding = None, iterations = 3):
#         super().__init__()
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.padding = padding or (kernel_size - 1) // 2
#         self.input_shape = (input_shape[0] + 2 * self.padding, input_shape[1] + 2 * self.padding)
#         self.in_capsules = in_capsules
#         self.in_cap_dim = in_cap_dim
#         self.out_capsules = out_capsules
#         self.out_cap_dim = out_cap_dim
#         self.iterations = iterations
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.W = nn.Parameter(torch.randn(1, 1, 1, kernel_size ** 2, in_capsules, out_capsules, in_cap_dim, out_cap_dim, device = device))
#         self.device = device
#         self.kernel_map, self.parent_size = self.__get_kernel_map(self.input_shape, self.kernel_size, self.stride)
#         self.children_per_parent = self.__group_children_by_parent(self.kernel_map)

#         nn.init.xavier_normal_(self.W)
#     def __init_c(self, kernel_map, parent_shape, in_capsules, out_capsules):
#         num_parents_per_child = kernel_map.sum(axis = 0, keepdims = True)
#         initial_c = kernel_map / (num_parents_per_child * out_capsules + 1e-6)
#         initial_c = initial_c[kernel_map == 1].reshape(*parent_shape, -1)[..., None, None]
#         initial_c = np.tile(initial_c, [1, 1, 1, in_capsules, out_capsules])[None]
#         return initial_c
#     def __update_c(self, b):
#         #N, ps, ps, ks * ks, in_caps, out_caps
#         bs = b.size(0)
#         ph = b.size(1)
#         pw = b.size(2)
#         ks2 = b.size(3)
#         in_caps = b.size(-2)
#         b = b.permute(0, 3, 4, 1, 2, 5)
#         b = b.reshape(bs, ks2, in_caps, -1)
#         c = F.softmax(b, dim = -1)
#         c = c.reshape(bs, ks2, in_caps, ph, pw, -1)
#         c = c.permute(0, 3, 4, 1, 2, 5)
#         return c
#     def __get_kernel_map(self, child_shape, ks, stride):
#         parent_h = (child_shape[0] - ks) // stride + 1
#         parent_w = (child_shape[1] - ks) // stride + 1
#         kernel_map = np.zeros((parent_h * parent_w, np.prod(child_shape)))
#         for r in range(parent_h):
#             for c in range(parent_w):
#                 p_idx = r * parent_w + c
#                 for i in range(ks):
#                     c_idx = r * stride * child_shape[1] + c * stride + i * child_shape[0]
#                     kernel_map[p_idx, c_idx : c_idx + ks] = 1
#         return kernel_map, (parent_h, parent_w)
#     def __group_children_by_parent(self, map):
#         return np.where(map)[1].reshape(map.shape[0], -1)
#     def forward(self, x):
#         x = F.pad(x, (0, 0, 0, 0, self.padding, self.padding, self.padding, self.padding))
#         kernel_map, parent_size = self.kernel_map, self.parent_size
#         children_per_parent = self.children_per_parent

#         x_unroll = x.view(x.shape[0], -1, self.in_capsules, self.in_cap_dim)
#         tile = x_unroll[:, children_per_parent]
#         tile = tile.view(x.shape[0], *parent_size, self.kernel_size ** 2, self.in_capsules, self.in_cap_dim).unsqueeze(-2).unsqueeze(-2)
#         u_hat = torch.matmul(tile, self.W).squeeze(-2)
#         u_hat_detach = u_hat.detach()
#         b = torch.zeros(x.shape[0], *parent_size, self.kernel_size ** 2, self.in_capsules, self.out_capsules)
#         for i in range(self.iterations - 1):
#             if i == 0:
#                 c = torch.Tensor(self.__init_c(kernel_map, parent_size, self.in_capsules, self.out_capsules), device = self.device)
#             else:
#                 c = self.__update_c(b)
#             # N x ps x ps x o x out_dim
#             s = torch.sum(c.unsqueeze(-1) * u_hat_detach, dim = [3, 4])
#             v = squash(s)
#             b += torch.sum(u_hat_detach * v.unsqueeze(-3).unsqueeze(-3), dim = -1)

#         c = self.__update_c(b)
#         s = torch.sum(c.unsqueeze(-1) * u_hat, dim = [3, 4])
#         v = squash(s)
#         return v

# class ConvTranposeCapsule(nn.Module):
#     def __init__(self, input_shape, in_capsules, in_cap_dim, out_capsules, out_cap_dim, kernel_size, stride, padding = 0, iterations = 3):
#         super().__init__()
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.padding = padding
#         self.input_shape = input_shape
#         self.in_capsules = in_capsules
#         self.in_cap_dim = in_cap_dim
#         self.out_capsules = out_capsules
#         self.out_cap_dim = out_cap_dim
#         self.iterations = iterations
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.W = nn.Parameter(torch.randn(in_cap_dim, out_capsules * out_cap_dim, kernel_size, kernel_size).to(device))
#         self.device = device
#         nn.init.xavier_normal_(self.W)

#     def forward(self, x):
#       in_shape = x.shape
#       x = x.permute(0, 3, 4, 1, 2)
#       x = x.reshape(-1, self.in_cap_dim, *self.input_shape)
#       x = F.conv_transpose2d(x, self.W, stride = self.stride, padding = self.padding)
#       x = x.view(in_shape[0], in_shape[-2], self.out_capsules, self.out_cap_dim, x.shape[2], x.shape[3])
#       u_hat = x.permute(0, 1, 4, 5, 2, 3)
      
#       b = torch.zeros(u_hat.shape[:-1]).to(self.device)
#       for _ in range(self.iterations - 1):
#         c = F.softmax(b, dim = -2)
#         s = torch.sum(u_hat * c.unsqueeze(-1), dim = 1)
#         v = squash(s)
#         b = b + torch.sum(u_hat * v.unsqueeze(1), dim = -1)
#       c = F.softmax(b, dim = -2)
#       s = torch.sum(u_hat * c.unsqueeze(-1), dim = 1)
#       v = squash(s)
#       return v


class ConvCapsule(nn.Module):
    def __init__(self, input_shape, in_capsules, in_cap_dim, out_capsules, out_cap_dim, kernel_size, stride, padding = 0, iterations = 3):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padding = padding
        self.input_shape = input_shape
        self.in_capsules = in_capsules
        self.in_cap_dim = in_cap_dim
        self.out_capsules = out_capsules
        self.out_cap_dim = out_cap_dim
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.stride = stride
        self.W = nn.Parameter(torch.randn(out_capsules * out_cap_dim, in_cap_dim, kernel_size, kernel_size, device = device))
        self.device = device
        nn.init.xavier_normal_(self.W)

    def forward(self, x):
      in_shape = x.shape
      x = x.permute(0, 3, 4, 1, 2)
      x = x.reshape(-1, self.in_cap_dim, *self.input_shape)
      x = F.conv2d(x, self.W, stride = self.stride, padding = self.padding)
      x = x.view(in_shape[0], in_shape[-2], self.out_capsules, self.out_cap_dim, x.shape[2], x.shape[3])
      u_hat = x.permute(0, 1, 4, 5, 2, 3)
      
      b = torch.zeros(u_hat.shape[:-1]).to(self.device)
      for _ in range(self.iterations - 1):
        c = F.softmax(b, dim = -2)
        s = torch.sum(u_hat * c.unsqueeze(-1), dim = 1)
        v = squash(s)
        b = b + torch.sum(u_hat * v.unsqueeze(1), dim = -1)
      c = F.softmax(b, dim = -2)
      s = torch.sum(u_hat * c.unsqueeze(-1), dim = 1)
      v = squash(s)
      return v

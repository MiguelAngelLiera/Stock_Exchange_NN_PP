import torch
import numpy as np
from torch.nn import Module
import torch.nn.functional as F

class Net(Module):
    def __init__(self, h, w):
        super(Net, self).__init__()
        self.c1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.f2 = torch.nn.Linear(32 * h * w, 5)

    def forward(self, x):
        x = self.c1(x)
        x = x.view(x.size(0), -1)
        x = self.f2(x)
        return x

def haha(a, b, c, d):
    p = [a.view(32, 1, 3, 3), b, c.view(5, 32 * 12 * 12), d]
    x = torch.randn(size=[8, 1, 12, 12], dtype=torch.float32)
    y = torch.randint(0, 5, [8])
    x = F.conv2d(x, p[0], p[1], 1, 1)
    x = x.view(x.size(0), -1)
    x = F.linear(x, p[2], p[3])
    loss = F.cross_entropy(x, y)
    return loss


if __name__ == '__main__':
    net = Net(12, 12)

    h = torch.autograd.functional.hessian(haha, tuple([_.view(-1) for _ in net.parameters()]))

    print(h)
    
    # Then we just need to fix tensors in h into a big matrix
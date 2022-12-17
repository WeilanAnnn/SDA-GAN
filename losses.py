import torch.nn as nn
import torch
from torch import autograd
from torch.nn import functional as F

class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super().__init__()
        self.eps = 1e-8
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x + self.eps)
        y = self.softmax(y + self.eps)
        #y = y + self.eps
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x + self.eps)
        y = self.log_softmax(y + self.eps)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class WassersteinDistance(nn.Module):
    def __init__(self, drift=0.001):
        super().__init__()
        self.drift = drift
        pass

    def forward(self, real_data, fake_data):
        return (torch.mean(fake_data) - torch.mean(real_data)
                + (self.drift * torch.mean(real_data ** 2)))


class JSDivergence1(KLDivergence):
    def __init__(self):
        super().__init__(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p_output, q_output):
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (self.KLDivLoss(log_mean_output, p_output) + self.KLDivLoss(log_mean_output, q_output)) / 2


class GradientLoss():
    def __init__(self, loss=nn.L1Loss(), n_scale=3):
        super(GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.criterion = loss
        self.n_scale = n_scale

    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def getloss(self, realIm,fakeIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm = self.downsample(fakeIm)
            realIm = self.downsample(realIm)
            grad_fx, grad_fy = self.grad_xy(fakeIm)
            grad_rx, grad_ry = self.grad_xy(realIm)
            loss += pow(4, i) * self.criterion(grad_fx, grad_rx) + self.criterion(grad_fy, grad_ry)
        return loss

def dark(img):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size // 2, patch_size // 2))
    dark = maxpool(0 - img[:, None, :, :, :])
    dark = dark.squeeze(2)

    return -dark
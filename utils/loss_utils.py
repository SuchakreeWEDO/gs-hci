#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, floor
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    ''' We apply the depth correlation loss every iteration with a
    patch size of 128 pixels and select 50% of all patches per iteration
    We choose λ_depth = 0.1 
    box_p = patch_size 128 ?
    p_corr = ratio of selected patches 0.5 ? '''

    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = floor(depth_src.shape[0] / box_p)
    num_box_w = floor(depth_src.shape[1] / box_p)
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p

    # Select the number of boxes based on hyperparameter p_corr
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p

    _loss = torch.tensor(0.0,device='cuda')
                         
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return _loss / n_corr

# by fabby bc SparseGS did not provide the 'pearson_depth_loss' function
# Tested to output the same r (stats) value as scipy.pearsonr up to 15 decimal points
# ref: https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_stats_py.py#L4492-L4829
def pearson_depth_loss(x,y):
    '''torch version of scipy.pearsonr => output range [-1,1]
    Positive correlations imply that as x increases, so does y.
    Negative correlations imply that as x increases, y decreases.'''

    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')

    if n < 2:
        raise ValueError('x and y must have length at least 2.')


    if (x == x[0]).all() or (y == y[0]).all():
        if (x == x[0]).all():
            print("monodepth array is constant; the correlation coefficient "
                    "is not defined.")
        if (y == y[0]).all():
            print("depth_map array is constant; the correlation coefficient "
                    "is not defined.")
        return 0

    xmean = torch.mean(x)
    ymean = torch.mean(y)

    xm = torch.sub(x, xmean)
    ym = torch.sub(y, ymean)

    normxm = torch.linalg.norm(xm)
    normym = torch.linalg.norm(ym)
    
    r = torch.matmul(xm/normxm, ym/normym)
    
    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)
    
    # correlation higher is good = score
    # but we want to minimize loss so we add minus
    return - r
import os
import numpy as np
# import torch.nn as nn
import torch.nn.functional as F

from scipy import stats
from torch.utils.tensorboard import SummaryWriter

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def loss_fn(y_pred, y):
    # loss_cuda = nn.L1Loss().cuda()
    # return loss_cuda(y_pred, y)
    return F.l1_loss(y_pred, y)

def measure(sq, q, sq_std):
    sq = np.reshape(np.asarray(sq), (-1,))
    sq_std = np.reshape(np.asarray(sq_std), (-1,))
    q = np.reshape(np.asarray(q), (-1,))

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    outlier_ratio = (np.abs(sq - q)>2*sq_std).mean()
    return (srocc, krocc, plcc, rmse, outlier_ratio)

def tb_write(writer, metric, epoch):
    SROCC, KROCC, PLCC, RMSE, OR = metric
    writer.add_scalar("SROCC", SROCC, epoch)
    writer.add_scalar("KROCC", KROCC, epoch)
    writer.add_scalar("PLCC", PLCC, epoch)
    writer.add_scalar("RMSE", RMSE, epoch)
    writer.add_scalar("OR", OR, epoch)
import os
import torch
import random
import numpy as np
import torch.nn.functional as F

from scipy import stats
from torch.utils.tensorboard import SummaryWriter

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y)

def measure(sq, q):
    sq = np.reshape(np.asarray(sq), (-1,))
    q = np.reshape(np.asarray(q), (-1,))

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    return (srocc, krocc, plcc, rmse)

def tb_write(writer, metric, epoch):
    SROCC, KROCC, PLCC, RMSE = metric
    writer.add_scalar("SROCC", SROCC, epoch)
    writer.add_scalar("KROCC", KROCC, epoch)
    writer.add_scalar("PLCC", PLCC, epoch)
    writer.add_scalar("RMSE", RMSE, epoch)

def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
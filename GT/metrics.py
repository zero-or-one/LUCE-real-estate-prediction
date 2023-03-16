import torch
import torch.nn as nn
import torch.nn.functional as F

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import mean_absolute_error


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    #MAE = mean_absolute_error(scores.detach().cpu().numpy().reshape(1,-1),\
    # targets.detach().cpu().numpy().reshape(1,-1))
    return MAE

def MSE(scores, targets):
    MSE = F.mse_loss(scores, targets)
    MSE = MSE.detach().item()
    return MSE
import torch.nn as nn
import torch.nn.functional as F 




def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1),
           F.softmax(teacher_scores/T, dim=1)) * (alpha * T * T) + \
           F.cross_entropy(y, labels) * (1. - alpha)

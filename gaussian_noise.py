import torch


def add_noise(x, noise_variance=0.1):
    
    noise = torch.zeros(x.shape[0],x.shape[1], x.shape[2], x.shape[3]).data.normal_(0, noise_variance).cuda()
    x = x + noise
    
    return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def nll_loss(output, target):
    return F.nll_loss(output, target)

def d_loss(real_logit, fake_logit):
    real_loss = F.binary_cross_entropy_with_logits(real_logit, 
                                                   torch.ones_like(real_logit))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logit, 
                                                   torch.zeros_like(fake_logit))
    return real_loss, fake_loss

def g_loss(fake_logit):
    fake_loss = F.binary_cross_entropy_with_logits(fake_logit, 
                                                   torch.ones_like(fake_logit))
    return fake_loss

def wasserstein_d_loss(real_logit, fake_logit):
    real_loss = -real_logit.mean()
    fake_loss = fake_logit.mean()
    return real_loss, fake_loss

def wasserstein_g_loss(fake_logit):
    fake_loss = -fake_logit.mean()
    return fake_loss

def gradient_penalty(d_model, real, fake, label=None):
    device = real.device
    batch_size = real.size(0)

    alpha = torch.randn(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real)
    alpha = alpha.to(device)
    interpolated = (alpha * real) + ((1-alpha) * fake)
    interpolated.requires_grad_()

    if label is not None:
        score = d_model(interpolated, label)
    else:
        score = d_model(interpolated)

    gradient = grad(score, interpolated, 
                    grad_outputs=torch.ones(score.size()).to(device),
                    create_graph=True)[0]

    penalty = ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()
    return penalty

# Collection of Loss Functions

import torch
import torch.nn.functional as F

def get_losses_fn(mode):
    if mode == 'gan':
        """
        Loss from original GANs paper - 2014 Ian Goodfellow
        """
        def d_loss_fn(real_logit, fake_logit):
            real_loss = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
            return real_loss, fake_loss

        def g_loss_fn(fake_logit):
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
            return fake_loss

    elif mode == 'lsgan':
        def d_loss_fn(real_logit, fake_logit):
            real_loss = F.mse_loss(real_logit, torch.ones_like(real_logit))
            fake_loss = F.mse_loss(fake_logit, torch.zeros_like(fake_logit))
            return real_loss, fake_loss

        def g_loss_fn(fake_logit):
            fake_loss = F.mse_loss(fake_logit, torch.ones_like(fake_logit))
            return fake_loss

    elif mode == 'wgan':
        """
        Wasserstein (Earth Mover) loss function - 2017
        """
        def d_loss_fn(real_logit, fake_logit):
            real_loss = -real_logit.mean()
            fake_loss = fake_logit.mean()
            return real_loss, fake_loss

        def g_loss_fn(fake_logit):
            fake_loss = -fake_logit.mean()
            return fake_loss

    elif mode == 'hinge_v1':
        def d_loss_fn(real_logit, fake_logit):
            real_loss = torch.max(1 - real_logit, torch.zeros_like(real_logit)).mean()
            fake_loss = torch.max(1 + fake_logit, torch.zeros_like(fake_logit)).mean()
            return real_loss, fake_loss

        def g_loss_fn(fake_logit):
            fake_loss = torch.max(1 - fake_logit, torch.zeros_like(fake_logit)).mean()
            return fake_loss

    elif mode == 'hinge_v2':
        def d_loss_fn(real_logit, fake_logit):
            real_loss = torch.max(1 - real_logit, torch.zeros_like(real_logit)).mean()
            fake_loss = torch.max(1 + fake_logit, torch.zeros_like(fake_logit)).mean()
            return real_loss, fake_loss

        def g_loss_fn(fake_logit):
            fake_loss = - fake_logit.mean()
            return fake_loss

    else:
        raise NotImplementedError

    return d_loss_fn, g_loss_fn
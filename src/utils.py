# Selection of helper functions

import torch
import torch.nn as nn

def use_gpu(GPU=True, device_idx=0):
    """

    """
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    return device

def fix_seed(seed=0):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

# From DL Tutorial 3. Seems to help improve brightness of CIFAR10 image
def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        assert None not in [channels, w, h], "Number of channels, width, height must be provided"
        x = x.view(x.size(0), channels, w, h)
    return x

# From DL Tutorial 3. Not sure the reasoning behind this initialization
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def param_count(model):
    """
    Returns number of model parameters
    """
    parms = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return parms


class NoOp(nn.Module):

    def __init__(self, *args, **keyword_args):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


def identity(x, *args, **keyword_args):
    return x


def _get_norm_func_2D(norm):  # 2d
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == None:
        return NoOp
    else:
        raise NotImplementedError


def _get_weight_norm_func(weight_norm):
    if weight_norm == 'spectral_norm':
        return nn.utils.spectral_norm
    elif weight_norm == 'weight_norm':
        return nn.utils.weight_norm
    elif weight_norm == None:
        return identity
    else:
        return NotImplementedError













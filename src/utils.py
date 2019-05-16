# Selection of helper functions

import torch

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














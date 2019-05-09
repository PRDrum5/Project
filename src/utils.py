# Selection of helper functions


# From DL Tutorial 3. Seems to help improve brightness of CIFAR10 image
def denorm(x, channels=None, w=None, h=None, resize = False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x
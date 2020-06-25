
def seed_rng(seed=0):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config():
    from dotenv import load_dotenv
    load_dotenv()


def cifar10_root():
    """
    Checks to see if cifar10 data is on disk, if not downloads it.
    :return: path to CIFAR10 data
    """
    import os
    load_config()
    path = os.environ.get('CIFARROOT', None)
    assert path is not None
    if not os.path.isdir(path):
        import torchvision
        torchvision.datasets.CIFAR10(path, download=True)
    return path


def numpy_suppress_scientific():
    import numpy as np
    np.set_printoptions(suppress=True)

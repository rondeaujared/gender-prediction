
def seed_rng(seed=0):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)


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


def sample_dataset(dataset, batch_size):
    """

    :param dataset:
    :param batch_size:
    :return: [Tensor: (batch_size, 3, height, width), Tensor: (batch_size)]
    """
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch_tensor = next(iter(dl))
    return batch_tensor

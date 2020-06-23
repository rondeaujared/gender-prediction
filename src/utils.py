
def seed_rng(seed=0):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_config():
    from dotenv import load_dotenv
    load_dotenv()
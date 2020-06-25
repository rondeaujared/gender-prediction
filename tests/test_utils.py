import unittest
import os


class ReproducibilityTestCase(unittest.TestCase):
    """
    See https://pytorch.org/docs/stable/notes/randomness.html
    """

    def test_numpy_rng(self):
        import numpy as np
        from src.utils import seed_rng
        seed_rng(1)
        arr = np.random.randint(10, size=5)
        self.assertTrue(
            np.all(np.array_equal(arr, np.array([5, 8, 9, 5, 0]))),
            f"{arr}"
        )

    def test_torch_rng(self):
        import torch
        from src.utils import seed_rng
        seed_rng(2)
        arr = torch.randint(low=0, high=10, size=(5,))
        self.assertTrue(
            torch.all(torch.eq(arr, torch.Tensor([8, 7, 1, 4, 8])))
        )

    def test_random_rng(self):
        import random
        from src.utils import seed_rng
        seed_rng(3)
        a = random.randint(0, 10)
        self.assertEqual(a, 3)


class ConfigTestCase(unittest.TestCase):

    def test_dataroot(self):
        from src.utils import load_config
        load_config()
        fail = ''
        self.assertFalse(os.path.isdir(fail))
        path = os.environ.get('DATAROOT', fail)
        self.assertTrue(os.path.isdir(path),
                        f"Environment variable $DATAROOT {path} is not set.")

    def test_cuda_devices(self):
        """
        Checks to see if Tensor can be pushed to GPU indexed at 0
        :return:
        """
        import torch
        for device in [torch.device('cpu'), torch.device('cuda:0')]:
            a = torch.Tensor([1, 2, 3]).to(device=device)
            self.assertEqual(device, a.device)


if __name__ == '__main__':
    unittest.main()

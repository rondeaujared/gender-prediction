import unittest


class VisualizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from src.utils import cifar10_root
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        # self.cifar10_root = cifar10_root()
        self.BATCH_SIZE = 8
        self.ds = CIFAR10(root=cifar10_root(), transform=ToTensor())

    def test_sample_dataset(self):
        from src.utils import sample_dataset
        images, labels = sample_dataset(self.ds, self.BATCH_SIZE)

        self.assertEqual(images.shape, (self.BATCH_SIZE, 3, 32, 32))
        self.assertEqual(labels.shape, (self.BATCH_SIZE,))

    def test_visualize_batch(self):
        from src.visualization import visualize_batch
        from src.utils import sample_dataset
        images, labels = sample_dataset(self.ds, self.BATCH_SIZE)
        visualize_batch(images, 'test_visualize_batch.png')


if __name__ == '__main__':
    unittest.main()

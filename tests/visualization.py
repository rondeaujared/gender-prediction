import unittest


class VisualizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from src.utils import cifar10_root
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        self.BATCH_SIZE = 8
        self.ds = CIFAR10(root=cifar10_root(), transform=ToTensor())

    def test_sample_dataset(self):
        from src.visualization import sample_dataset
        images, labels = sample_dataset(self.ds, self.BATCH_SIZE)

        self.assertEqual(images.shape, (self.BATCH_SIZE, 3, 32, 32))
        self.assertEqual(labels.shape, (self.BATCH_SIZE,))

    def test_visualize_batch(self):
        from src.visualization import visualize_batch
        from src.visualization import sample_dataset
        images, labels = sample_dataset(self.ds, self.BATCH_SIZE)
        visualize_batch(images, 'plots/test_visualize_batch.png')

    def test_draw_label_on_batch(self):
        from src.visualization import (visualize_batch,
                                       sample_dataset, draw_label_on_batch)
        image_tensor, label_tensor = sample_dataset(self.ds, self.BATCH_SIZE)
        image_tensor = draw_label_on_batch(image_tensor, label_tensor)
        visualize_batch(image_tensor, 'plots/test_draw_label_on_batch.png')

    def test_labels_to_colors(self):
        import numpy as np
        from src.visualization import labels_to_colors

        for labels in [np.array([1, 2, 2, 3]), np.array(['red', 'red', 'green', 'blue'])]:
            colors = labels_to_colors(labels)
            print(colors)
            self.assertEqual(len(np.unique(labels)), len(np.unique(colors)))

    def test_plot_2d(self):
        from src.visualization import plot_2d
        import numpy as np
        data = np.concatenate([np.random.normal(mu, 1, (5, 2)) for mu in [1, 10, 100]], axis=0)
        for labels, s in zip([np.array([1]*5 + [2]*5 + [3]*5), None], [1, 2]):
            plot_2d(data, labels, f'plots/test_plot_2d_{s}.png')


class ClusteringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_k_means(self):
        import numpy as np
        from src.utils import numpy_suppress_scientific
        numpy_suppress_scientific()
        data = np.concatenate([np.random.normal(mu, 1, (5, 2)) for mu in [1, 10, 100]], axis=0)
        print(data, data.shape)


if __name__ == '__main__':
    unittest.main()

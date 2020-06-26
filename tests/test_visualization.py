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
            self.assertEqual(len(np.unique(labels)), len(np.unique(colors)))

    def test_plot_2d(self):
        from src.visualization import plot_2d
        import numpy as np
        data = np.concatenate([np.random.normal(mu, 1, (5, 2)) for mu in [1, 10, 100]], axis=0)
        for labels, s in zip([np.array([1]*5 + [2]*5 + [3]*5), None], [1, 2]):
            plot_2d(data, labels, f'plots/test_plot_2d_{s}.png')

    def test_scatterplot_images(self):
        import numpy as np
        from torchvision.transforms import ToPILImage
        from src.visualization import scatterplot_images, sample_dataset
        from src.utils import seed_rng
        seed_rng(0)
        images, labels = sample_dataset(self.ds, 100)
        labels = labels.numpy()
        trans = ToPILImage()
        images = [trans(img) for img in images]
        lbl2x0 = {lbl: 48*ix+32 for lbl, ix in enumerate(np.unique(labels))}
        cnt_lbl = {lbl: 0 for lbl in np.unique(labels)}
        x, y = [], []
        for img, lbl in zip(images, labels):
            x.append(lbl2x0[lbl])
            y.append(cnt_lbl[lbl]*48+32)
            cnt_lbl[lbl] += 1
        data = np.stack([x, y], axis=1)
        scatterplot_images(data, images, f'plots/test_scatterplot_images.png')

    def test_plot_weight_dist(self):
        import torch
        import numpy as np
        from torchvision.models import resnet18
        from src.visualization import plot_weight_dist

        for trained in [True, False]:
            with torch.no_grad():

                model = resnet18(pretrained=trained)
                model.eval()
                l_name, l_x = [], []

                for n, weight in model.named_parameters():
                    w = np.array(weight.detach().flatten())
                    l_name.append(f"{n}\nmean={w.mean():.3f}\nstd={w.std():.3f}")
                    l_x.append(w)

            plot_weight_dist(l_name, l_x, f'plots/test_plot_weight_dist_{trained}.png')

    def test_plot_activation_dist(self):
        import torch
        import numpy as np
        from torchvision.models import resnet18
        from src.visualization import plot_weight_dist, sample_dataset

        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        for trained in [True, False]:
            with torch.no_grad():
                model = resnet18(pretrained=trained)
                model.eval()
                l_name, l_x = [], []
                for n, weight in model.named_modules():
                    weight.register_forward_hook(get_activation(n))
                images, labels = sample_dataset(self.ds, self.BATCH_SIZE)
                model(images)
                for k, v in activations.items():
                    w = np.array(v.detach().flatten())
                    l_name.append(f"{k}\nmean={w.mean():.3f}\nstd={w.std():.3f}")
                    l_x.append(w)
            plot_weight_dist(l_name, l_x, f'plots/test_plot_activation_dist_{trained}.png')

    def test_plot_gradient_dist(self):
        import torch
        import numpy as np
        from torchvision.models import resnet18
        from src.visualization import plot_weight_dist, sample_dataset

        gradients = {}
        grad_norm = {}

        def get_gradients(name):
            def hook(model, grad_input, grad_output):
                gradients[name] = grad_output
                grad_norm[name] = grad_output[0].norm()
            return hook

        loss_fn = torch.nn.CrossEntropyLoss()
        for trained in [True, False]:
            model = resnet18(pretrained=trained)
            model.train()
            l_name, l_x = [], []
            for n, weight in model.named_modules():
                weight.register_backward_hook(get_gradients(n))
            images, labels = sample_dataset(self.ds, self.BATCH_SIZE)
            out = model(images)
            err = loss_fn(out, labels)
            err.backward()
            for k, v in gradients.items():
                w = np.array(v[0].detach().flatten())
                norm = grad_norm[k].numpy()
                l_name.append(f"{k}\nmean={w.mean():.3f}\nstd={w.std():.3f}\nnorm:{norm:.3f}")
                l_x.append(w)
            plot_weight_dist(l_name, l_x, f'plots/test_plot_gradient_dist_{trained}.png')


class ClusteringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_k_means(self):
        import numpy as np
        from src.utils import numpy_suppress_scientific
        from src.visualization import k_means, plot_2d
        numpy_suppress_scientific()
        data = np.concatenate([np.random.normal(mu, 1, (5, 2)) for mu in [1, 10, 100]], axis=0)
        for k in [2, 3]:
            out = k_means(data, n_clusters=k)
            labels, centers = out['labels'], out['centers']
            plot_2d(data, labels, f'plots/test_k_means_{k}.png')

    def test_k_means2(self):
        import numpy as np
        np.set_printoptions(suppress=True)
        from src.visualization import k_means
        data = np.concatenate([np.random.normal(mu, 1, (5, 101)) for mu in [1, 10, 100]], axis=0)
        out = k_means(data, n_clusters=2)
        # print(f"centroids: {out['centers']}")


if __name__ == '__main__':
    unittest.main()

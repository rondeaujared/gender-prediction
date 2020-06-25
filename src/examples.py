def stack_outputs_and_plot():
    import torch
    import numpy as np
    from torchvision.models.resnet import resnet18
    from src.utils import cifar10_root
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    model = resnet18(pretrained=True).cuda()
    model.eval()
    dl = DataLoader(CIFAR10(root=cifar10_root(), transform=ToTensor()), batch_size=8, shuffle=False)
    l_outputs = []
    l_labels = []
    dl_iter = iter(dl)
    with torch.no_grad():
        for ix in range(4):
            images, labels = next(dl_iter)
            images = images.cuda()
            l_outputs.append(model(images).cpu().numpy())
            l_labels.append(labels.numpy())
    outputs = np.concatenate(l_outputs, axis=0)
    labels = np.concatenate(l_labels, axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(outputs)
    data = pca.transform(outputs)

    from src.visualization import plot_2d
    plot_2d(data, labels, f'plots/stack_outputs_and_plot.png')

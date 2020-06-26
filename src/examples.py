

def stack_outputs_and_plot():
    """
    (1)  Perform forward pass w/ CNN on batches of CIFAR10
    (2)  Collect input images, labels, model output tensors
    (3)  Map outputs to 2D with PCA
    (4)  Plot input Images in 2D output space
    :return:
    """
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.models.resnet import resnet18
    from src.utils import cifar10_root
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor, ToPILImage
    from torch.utils.data import DataLoader

    model = resnet18(pretrained=True).cuda()
    model.eval()
    dl = DataLoader(CIFAR10(root=cifar10_root(), transform=ToTensor()), batch_size=8, shuffle=False)
    l_outputs = []
    l_labels = []
    l_images = []
    dl_iter = iter(dl)
    trans = ToPILImage()
    with torch.no_grad():
        for ix in range(4):
            images, labels = next(dl_iter)
            batch = images.cuda()
            l_outputs.append(model(batch).cpu().numpy())
            l_labels.append(labels.numpy())
            l_images.extend([trans(img) for img in images])
    outputs = np.concatenate(l_outputs, axis=0)
    labels = np.concatenate(l_labels, axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(outputs)
    data = pca.transform(outputs)

    from src.visualization import scatterplot_images
    scatterplot_images(data, l_images, f'plots/scatterplot_images.png')

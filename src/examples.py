
def scatterplot_images(data, images, savefig=''):
    """

    :param data:    (N, 2) nparray
    :param images:  [*PIL.Image]
    :param savefig: str or None path to save
    :return:
    """
    import matplotlib.pyplot as plt
    import io
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    def getImage(_data):
        return OffsetImage(plt.imread(_data))
    fig, ax = plt.subplots()
    x, y = data[:, 0], data[:, 1]
    ax.scatter(x, y)

    for x0, y0, image in zip(x, y, images):
        b = io.BytesIO()
        image.save(b, "png")
        b.seek(0)
        ab = AnnotationBbox(getImage(b), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.show()
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()


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
    scatterplot_images(data, l_images, f'plots/scatterplot_images.png')

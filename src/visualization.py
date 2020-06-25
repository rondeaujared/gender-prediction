def visualize_batch(images_tensor, savefig='', nrows=5):
    """
    :param nrows:
    :param images_tensor: Shape (batch_size, 3, height, width) tensor
    :param savefig: If provided, saves plt figure to savefig.png; otherwise calls plt.show.
    :return:
    """
    import torchvision
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(images_tensor, nrow=nrows)
    plt.imshow(grid_img.permute(1, 2, 0))  # plt wants RGB channels as last dim
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()


def sample_dataset(dataset, batch_size):
    """

    :param dataset:
    :param batch_size:
    :return: (Tensor: (batch_size, 3, height, width), Tensor: (batch_size))
    """
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    images, labels = next(iter(dl))
    return images, labels


def draw_label_on_batch(images_tensor, labels):
    """

    :param images_tensor: Tensor: (batch_size, 3, height, width)
    :param labels: iterable: [*batch_size]
    :return: drawn on images_tensor
    """
    from torch import stack
    from torchvision.transforms import ToPILImage, ToTensor
    from PIL import Image, ImageDraw, ImageFont
    font_root = "/usr/share/fonts/truetype/freefont/Free"
    size = max(int(images_tensor.shape[2]*0.1), 10)
    font = ImageFont.truetype(f"{font_root}Serif.ttf", size)
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    tensors = []
    for img, lbl in zip(images_tensor, labels):
        img = to_pil(img)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"{lbl}", fill="red", font=font)
        tensors.append(to_tensor(img))
    out = stack(tensors, dim=0)
    return out


def labels_to_colors(labels):
    """
    Note: maximum of 10 unique labels supported.
    See https://matplotlib.org/3.1.0/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py

    :param labels: nparray of scalers such as [1, 2, 2] or ['cat', 'dog', 'cow']
    :return: list of colors s.t. each unique label is mapped to a color
    """
    import matplotlib.colors as mcolors
    import numpy as np
    lbl2clr = {lbl: clr for lbl, clr in zip(np.unique(labels), mcolors.TABLEAU_COLORS)}
    return [lbl2clr[lbl] for lbl in labels]


def plot_2d(data, labels=None, savefig=''):
    """

    :param labels:
    :param data: nparray (N, M)
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if labels is None:
        colors = ['r' for _ in data[:, 0]]
    else:
        colors = labels_to_colors(labels)
    print(f"colors: {colors}")
    ax.scatter(data[:, 0], data[:, 1], c=colors)
    ax.set_xlabel(f"x axis")
    ax.set_ylabel(f"y axis")
    ax.set_title(f"plot_2d")
    ax.grid(True)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()

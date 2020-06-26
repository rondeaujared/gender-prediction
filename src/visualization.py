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
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if labels is None:
        colors = ['r' for _ in data[:, 0]]
    else:
        colors = labels_to_colors(labels)
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


def k_means(X, n_clusters):
    """
    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    :param X:
    :param n_clusters:
    :return: {'labels': 1-D iterable of int labels, 'centers': np array of cluster centers}
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return {'labels': kmeans.labels_, 'centers': kmeans.cluster_centers_}


def scatterplot_images(data, images, savefig=''):
    """
    :param data:    (N, 2) nparray
    :param images:  [*PIL.Image]
    :param savefig: str or None path to save
    :return:
    """
    import matplotlib.pyplot as plt
    import io
    import numpy as np
    import math
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    def getImage(_data):
        return OffsetImage(plt.imread(_data))

    x, y = data[:, 0], data[:, 1]
    max_x, max_y = np.max(x), np.max(y)
    DPI = 96  # See http://auctionrepair.com/pixels.html
    width = math.ceil(max_x / DPI)
    height = math.ceil(max_y / DPI)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.tight_layout()
    ax.scatter(x, y)

    for x0, y0, image in zip(x, y, images):
        b = io.BytesIO()
        image.save(b, "png")
        b.seek(0)
        ab = AnnotationBbox(getImage(b), (x0, y0), frameon=False)
        ax.add_artist(ab)
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()


def plot_weight_dist(l_name, l_x, savefig=''):
    import matplotlib.pyplot as plt
    ncol = (len(l_name)-1) // 5 + 1
    nrow = 5
    fig, axs = plt.subplots(ncol, nrow, figsize=(nrow*3, ncol*2))
    if len(l_name) == 1:
        axs = [axs]
    fig.tight_layout()
    for ix, (name, x) in enumerate(zip(l_name, l_x)):
        row = ix//5
        col = ix % 5
        axs[row, col].set_xlabel(name)
        axs[row, col].hist(x)
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
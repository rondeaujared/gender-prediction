def visualize_batch(images_tensor, savefig='', nrows=5):
    """
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


def draw_label_on_batch(images_tensor, labels_tensor):
    """

    :param images_tensor: Tensor: (batch_size, 3, height, width)
    :param labels_tensor: Tensor: (batch_size)
    :return:
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
    for img, lbl in zip(images_tensor, labels_tensor):
        img = to_pil(img)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"{lbl}", fill="red", font=font)
        tensors.append(to_tensor(img))
    out = stack(tensors, dim=0)
    return out

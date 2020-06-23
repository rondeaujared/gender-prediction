def visualize_batch(batch_tensor, savefig=''):
    """
    :param batch_tensor: Shape (batch_size, 3, height, width) tensor
    :param savefig: If provided, saves plt figure to savefig.png; otherwise calls plt.show.
    :return:
    """
    import torchvision
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))  # plt wants RGB channels as last dim
    if savefig:
        plt.savefig(f'{savefig}')
    else:
        plt.show()


def gender_estimation():
    import os
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.optim import SGD, Adam
    from torch.nn import KLDivLoss, L1Loss, NLLLoss, CrossEntropyLoss
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms
    from torchvision.models.resnet import resnet18
    from src.utils import load_config
    from src.datasets import unpickle_imdb, ImdbDataset
    load_config()
    device = torch.device('cuda')
    imdb_root = os.environ['IMDB_ROOT']
    df = unpickle_imdb(f"{imdb_root}/imdb.pickle")
    trans = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])
    ds = ImdbDataset(root=imdb_root, df=df, transform=trans)
    tr, val = random_split(ds, [len(ds)-len(ds)//10, len(ds)//10])
    loss_fn = CrossEntropyLoss()
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optim = Adam(model.parameters(), lr=3e-4)
    tr_dl = DataLoader(tr, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    tr_log, val_log = {}, {}

    def log_epoch(preds, labels, loss, log):
        _, pred_class = torch.max(preds.data, 1)
        log[epoch].append({
            'loss': loss.item(),
            'count': labels.size(0),
            'correct': (pred_class == labels).sum().item(),
            'tp': ((pred_class == 1) & (labels == 1)).sum().item(),
            'tn': ((pred_class == 0) & (labels == 0)).sum().item(),
            'fp': ((pred_class == 1) & (labels == 0)).sum().item(),
            'fn': ((pred_class == 0) & (labels == 1)).sum().item(),
            'cnt_p': (labels == 1).sum().item(),
            'cnt_n': (labels == 0).sum().item(),
        })

    def print_log_epoch(_e, log, pretext=''):
        epoch_loss = [x['loss'] for x in log[_e]]
        sum_loss = sum(epoch_loss)
        cnt_loss = len(epoch_loss)
        avg_loss = sum_loss/cnt_loss
        print(f"{pretext}Epoch {_e}: Total Loss={sum_loss}\tAvg Loss={avg_loss}\tNum Batches={cnt_loss}")

        e_cnt = [x['count'] for x in log[_e]]
        e_correct = [x['correct'] for x in log[_e]]
        e_acc = sum(e_correct) / sum(e_cnt)
        print(f"{pretext}Epoch {_e}: Total Cnt={sum(e_cnt)}\tTotal Cor={sum(e_correct)}\tAcc={e_acc}")

        tp_cnt = sum([x['tp'] for x in log[_e]])
        tn_cnt = sum([x['tn'] for x in log[_e]])
        fp_cnt = sum([x['fp'] for x in log[_e]])
        fn_cnt = sum([x['fn'] for x in log[_e]])
        p_cnt = sum([x['cnt_p'] for x in log[_e]])
        n_cnt = sum([x['cnt_n'] for x in log[_e]])
        print(f"{pretext}Epoch {_e}: TP={tp_cnt}\tTN={tn_cnt}\tFP={fp_cnt}\tFN={fn_cnt}\tP cnt={p_cnt}\tN cnt={n_cnt}")

    def _epoch(train):
        if train:
            dl = tr_dl
            model.train()
        else:
            dl = val_dl
            model.eval()

        for ix, (img, label) in enumerate(dl):
            img = img.to(device=device)
            labels = label.to(device=device, dtype=torch.int64)
            preds = model(img)
            loss = loss_fn(preds, labels)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
                log_epoch(preds, labels, loss, tr_log)
            else:
                log_epoch(preds, labels, loss, val_log)

    epoch = 0
    for i in range(1):
        tr_log[epoch] = []
        _epoch(True)
        print_log_epoch(epoch, tr_log)
        epoch += 1
    val_log[epoch] = []
    _epoch(False)
    print_log_epoch(epoch, val_log, pretext='VAL::')


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


if __name__ == '__main__':
    gender_estimation()

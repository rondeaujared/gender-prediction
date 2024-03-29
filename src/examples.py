import os
import datetime
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import KLDivLoss, L1Loss, NLLLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet50
from torch.utils.tensorboard import SummaryWriter
from src.utils import load_config
from src.datasets import unpickle_imdb, ImdbDataset
from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD
from src import LOG_DIR, IMDB_ROOT


class GenderTrainer(object):
    def __init__(self, print_every_iters=50):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dtype = torch.float32
        self.savedir = f"{LOG_DIR}"
        self.e = 0
        self.train_steps = 0
        self.eval_steps = 0
        self.sw = SummaryWriter()

        self.print_every_iters = print_every_iters
        self.trans = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = ImdbDataset(root=IMDB_ROOT, df=unpickle_imdb(f"{IMDB_ROOT}/imdb.pickle"),
                         transform=self.trans)
        tr_ds, val_ds = random_split(ds, [len(ds)-len(ds)//10, len(ds)//10])
        self.tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
        self.val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        self.loss_fn = CrossEntropyLoss(reduction='mean')

        self.model = resnet18(pretrained=True)
        self.model.to(device=self.device, dtype=self.dtype)
        self.optim = Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-6)

    def _step(self, images, labels, train: bool):
        images = images.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=torch.int64)
        output = self.model(images)
        loss = self.loss_fn(output, labels)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.train_steps += 1
        else:
            self.eval_steps += 1
        with torch.no_grad():
            log = self._evaluate_step(output, labels, loss.item())

        return log

    def _epoch(self, train: bool):
        dl = self.tr_dl if train else self.val_dl
        loss_sum = 0
        step_cnt = 0
        logs = []
        for ix, (images, labels) in enumerate(dl):
            log = self._step(images, labels, train)
            loss = log['loss']
            logs.append(log)
            if ix % self.print_every_iters == 0:
                self.sw.add_scalar('loss/batch/train', loss, self.train_steps) if train \
                    else self.sw.add_scalar('loss/batch/valid', loss, self.eval_steps)
            loss_sum += loss
            step_cnt += 1
        elog = self._evaluate_epoch(logs)
        return elog

    def train(self, n_epochs: int):
        best_eval = np.inf
        for _ in range(n_epochs):
            self.model.train()
            tr_elog = self._epoch(train=True)
            self._tensorboard_log(self.sw, tr_elog, True, self.e)
            self.model.eval()
            with torch.no_grad():
                val_elog = self._epoch(train=False)
                self._tensorboard_log(self.sw, val_elog, False, self.e)
                score = val_elog['loss']
                if score < best_eval:
                    print(f"New best score: {score:.5f} beats {best_eval:.5f}")
                    self.save_weights(self.model, self.savedir)
                    best_eval = score

            self.e += 1

    @staticmethod
    def save_weights(model, savedir, prefix='', suffix=''):
        time = datetime.datetime.now()
        s = f"{prefix}_{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}_{suffix}.pth"
        fname = f"{savedir}/{s}"
        print(f"Saving to: {fname}")
        torch.save(model.state_dict(), fname)

    @staticmethod
    def _evaluate_step(output, labels, loss):
        _, pred_class = torch.max(output.data, 1)
        cnt = labels.size(0)
        correct = (pred_class == labels).sum().item()
        acc = correct / cnt

        def _cm(pred_match, labels_match):
            return ((pred_class == pred_match) & (labels == labels_match)).sum().item()
        log = {
            'count': cnt,
            'correct': correct,
            'acc': acc,
            'tp': _cm(1, 1),
            'tn': _cm(0, 0),
            'fp': _cm(1, 0),
            'fn': _cm(0, 1),
            'cnt_p': (labels == 1).sum().item(),
            'cnt_n': (labels == 0).sum().item(),
            'loss': loss,
        }
        return log

    @staticmethod
    def _evaluate_epoch(logs):
        sums = defaultdict(lambda: 0)
        for log in logs:
            for k, v in log.items():
                sums[k] += v
        avgs = {
            'acc': sums['correct'] / sums['count'],
            'tpr': sums['tp'] / (sums['tp'] + sums['fn']),
            'tnr': sums['tn'] / (sums['tn'] + sums['fp']),
            'fpr': sums['fp'] / (sums['fp'] + sums['tn']),
            'fnr': sums['fn'] / (sums['fn'] + sums['tp']),
            #  Divide loss by number of forward passes to get mean loss per batch.
            'loss': sums['loss'] / len(logs)
        }
        return avgs

    @staticmethod
    def _tensorboard_log(sw, log, train, epoch):
        log = {**log}
        log['loss/epoch'] = log['loss']
        log.pop('loss')
        for k, v in log.items():
            key = f'{k}/train' if train else f'{k}/valid'
            sw.add_scalar(key, v, epoch)


def gender_estimation(weights=None):
    load_config()
    device = torch.device('cuda')
    imdb_root = os.environ['IMDB_ROOT']
    df = unpickle_imdb(f"{imdb_root}/imdb.pickle")
    savedir = f"{os.environ['LOG_DIR']}"
    trans = transforms.Compose([
        # transforms.Resize(72),
        #transforms.RandomCrop(64),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = ImdbDataset(root=imdb_root, df=df, transform=trans)
    print(f"Loaded ds with {len(ds)} items.")
    tr, val = random_split(ds, [len(ds)-len(ds)//10, len(ds)//10])
    loss_fn = CrossEntropyLoss()

    #model = resnet50(pretrained=True)
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    """
    from src.simclr import ResNetSimCLR
    model = ResNetSimCLR('resnet50', 64)
    #if weights:
    #    model.load_state_dict(torch.load(weights))

    model.projector = nn.Sequential(
        nn.Linear(model.n_features, model.n_features, bias=False),
        nn.ReLU(),
        nn.Linear(model.n_features, 2, bias=False)
    )
    for param in model.encoder.parameters():
        param.requires_grad = False
    """
    model.to(device)
    optim = Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
    tr_dl = DataLoader(tr, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    tr_log, val_log = {}, {}

    def untrans_display(im):
        std, mean = torch.as_tensor(IMAGENET_STD), torch.as_tensor(IMAGENET_MEAN)
        if mean.ndim == 1: mean = mean[:, None, None]
        if std.ndim == 1: std = std[:, None, None]
        im.mul_(std).add_(mean)
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()

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
            # _, preds = model(img)
            loss = loss_fn(preds, labels)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
                log_epoch(preds, labels, loss, tr_log)
            else:
                log_epoch(preds, labels, loss, val_log)

    def _save_weights(prefix='', suffix=''):
        time = datetime.datetime.now()
        s = f"{prefix}_{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}_{suffix}.pth"
        fname = f"{savedir}/{s}"
        print(f"Saving to: {fname}")
        torch.save(model.state_dict(), fname)

    epoch = 0
    for i in range(20):
        tr_log[epoch] = []
        _epoch(True)
        print_log_epoch(epoch, tr_log)

        val_log[epoch] = []
        _epoch(False)
        print_log_epoch(epoch, val_log, pretext='VAL::')
        epoch += 1
        _save_weights(prefix=f'long')


def gender_inference(root, weights):
    import os
    from src import LOG_DIR, FONT_ROOT
    from src.face_detection import (txt_to_preds, extract_faces)
    from src.gender_estimation import (gender_predict, gender_txt_to_preds, gender_preds_to_txt, draw_genders)

    path = extract_faces(root)
    l_faces = txt_to_preds(path)
    path_face = [(path, dict(face)) for path, flist in l_faces for face in flist]
    print(path_face)
    l_genders = gender_predict(path_face, weights)
    print(l_genders)
    log = os.path.join(f"{LOG_DIR}", "images_gender_preds.txt")
    gender_preds_to_txt(l_genders, log)
    l_genders = gender_txt_to_preds(log)
    draw_genders(l_genders, 'tmp/drawn_genders')


def gender_activation_inference(root, weights):
    import os
    from src import LOG_DIR, FONT_ROOT
    from src.face_detection import (txt_to_preds, extract_faces)
    from src.gender_estimation import (gender_activations, cluster)

    path = extract_faces(root)
    l_faces = txt_to_preds(path)
    path_face = [(path, dict(face)) for path, flist in l_faces for face in flist]
    print(path_face)
    log = gender_activations(path_face, weights)
    cluster(log, f'tmp/gender_scatterplot_images.png')


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
    #_weights = f'/mnt/fastdata/SavedModels/resnet18_gender.pth'  # Good weights trained on 64x64 images
    _weights = f'/home/jrondeau/PycharmProjects/cnn-training/src/runs/Jul05_19-54-43_jrondeau-desktop/checkpoints/model.pth'
    # _path = '/mnt/fastdata/anno-ai/gender/streetview'
    # _path = '/mnt/fastdata/challenging-binary-age/adult/adult-hat'
    # _path = '/mnt/fastdata/appa-real-edited/valid'
    #_path = '/mnt/fastdata/web_crawler/flickr/teen/pchild'
    # gender_estimation()
    # gender_inference(_path, _weights)
    #gender_activation_inference(_path, _weights)
    trainer = GenderTrainer()
    trainer.train(n_epochs=5)

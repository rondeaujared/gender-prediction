import os
import datetime
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import KLDivLoss, L1Loss, NLLLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.resnet import resnet18
from PIL import Image, ImageDraw, ImageFont
from src.datasets import unpickle_imdb, ImdbDataset
from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD
from src import LOG_DIR, IMDB_ROOT, FONT_ROOT
from src.abstractions.training import AbstractTrainer


class GenderTrainer(AbstractTrainer):

    def log_step(self, output, labels, loss) -> dict:
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

    def log_epoch(self, logs):
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

    def tensorboard_log(self, log, train: bool):
        log = {**log}
        log['loss/epoch'] = log['loss']
        log.pop('loss')
        for k, v in log.items():
            key = f'{k}/train' if train else f'{k}/valid'
            self.writer.add_scalar(key, v, self.e)

    def score_fn(self, log, score) -> (bool, float):
        if score:
            return log['loss'] < score, log['loss']
        else:
            return True, log['loss']


class MyScheduler(object):
    def __init__(self, optim, **kwargs):
        self.warmup_steps = kwargs.pop('warmup_steps')
        scheduler_class = kwargs.pop('scheduler_class')
        self.scheduler = scheduler_class(optim, **kwargs)
        self.step_cnt = 1

    def step(self):
        if self.step_cnt > self.warmup_steps:
            self.scheduler.step()
        self.step_cnt += 1

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def train_gender_imdb():
    model = resnet18(pretrained=True)
    trans = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = ImdbDataset(root=IMDB_ROOT, df=unpickle_imdb(f"{IMDB_ROOT}/imdb.pickle"),
                     transform=trans)
    tr_ds, val_ds = random_split(ds, [len(ds) - len(ds) // 10, len(ds) // 10])
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    loss_fn = CrossEntropyLoss(reduction='mean')
    optim = Adam
    optim_kwargs = {
        'lr': 3e-4,
        'weight_decay': 1e-6,
    }
    scheduler = MyScheduler
    scheduler_kwargs = {
        'warmup_steps': len(tr_dl)*2,
        'scheduler_class': CosineAnnealingLR,
        'T_max': len(tr_dl),
        'eta_min': 0,
        'last_epoch': -1,
    }
    trainer = GenderTrainer(model, tr_dl, val_dl, loss_fn,
                            optim=optim, optim_kwargs=optim_kwargs,
                            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
    trainer.train(100)


if __name__ == '__main__':
    train_gender_imdb()


def preprocess(path, face, trans):
    im = Image.open(path).convert("RGB")
    x1, y1, width, height = face['x1'], face['y1'], face['width'], face['height']
    x2, y2 = x1 + width, y1 + height
    margin = 0.4
    x1 = int(max(x1 - width * margin, 0))
    y1 = int(max(y1 - height * margin, 0))
    x2 = int(min(x2 + width * margin, im.size[0]))
    y2 = int(min(y2 + height * margin, im.size[1]))
    _b = (x1, y1, x2, y2)
    im = im.crop(box=_b)
    return trans(im)


def gender_txt_to_preds(log_path):
    with open(log_path, 'r') as f:
        lines = f.read().splitlines()
        p2f = {}
        for line in lines:
            path, x1, y1, width, height, gender = line.split(' ')
            if path not in p2f:
                p2f[path] = []
            p2f[path].append({
                'x1': int(x1),
                'y1': int(y1),
                'width': int(width),
                'height': int(height),
                'gender': int(gender),
            })
        file_preds = [(k, v) for k, v in p2f.items()]
    return file_preds


def draw_genders(file_preds, savefig):
    os.makedirs(savefig, exist_ok=True)
    out = []
    for path, preds in file_preds:
        img = Image.open(path)
        draw = ImageDraw.Draw(img)

        for pred in preds:
            x1, y1 = pred['x1'], pred['y1']
            x2, y2 = x1 + pred['width'], y1 + pred['height']
            # gender = pred['gender']
            gender = 'm' if pred['gender'] > 0 else 'wo'
            size = max(pred['width'], 10)
            font = ImageFont.truetype(f"{FONT_ROOT}Serif.ttf", size)
            draw.rectangle((x1, y1, x2, y2), outline='red', width=1)
            draw.text((x1-size, y1-size), f"{gender}", fill="red", font=font)
        save_path = f"{savefig}/{path[path.rfind('/') + 1:]}"
        out.append((save_path, preds))
        img.save(save_path)
    return out


def gender_preds_to_txt(file_preds, log_path):
    with open(log_path, 'w') as f:
        out = [f"{path} {pred['x1']} {pred['y1']} {pred['width']} {pred['height']} {pred['gender']}\n"
               for path, pred in file_preds]
        f.writelines(out)


def _model_init(weights):
    device = torch.device('cuda')
    trans = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if weights:
        model.load_state_dict(torch.load(weights))
    model.to(device)
    model.eval()
    return device, trans, model


def gender_predict(path_face, weights):
    device, trans, model = _model_init(weights)
    l_preds = []
    with torch.no_grad():
        for path, face in path_face:
            img = preprocess(path, face, trans).unsqueeze(0).to(device=device)
            preds = model(img)
            _, pred_class = torch.max(preds.data, 1)
            l_preds.append((path, {**face, 'gender': pred_class.item()}))
    return l_preds


def gender_activations(path_face, weights):
    device, trans, model = _model_init(weights)
    log = []
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    for n, weight in model.named_modules():
        weight.register_forward_hook(get_activation(n))

    with torch.no_grad():
        for path, face in path_face:
            img = preprocess(path, face, trans).unsqueeze(0).to(device=device)
            preds = model(img)
            _acts = {}
            for k, v in activations.items():
                w = np.array(v.cpu().detach())
                _acts[k] = w[0].flatten()
            _, pred_class = torch.max(preds.data, 1)
            log.append({
                'path': path,
                'face': {**face},
                'gender': pred_class.item(),
                'activation': _acts,
            })

    return log


def gender_analyze(weights, dset):
    device, trans, model = _model_init(weights)
    dl = DataLoader(dset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    log = []
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    for n, weight in model.named_modules():
        weight.register_forward_hook(get_activation(n))

    with torch.no_grad():
        for ix, (img, label, path) in enumerate(dl):
            img = img.to(device=device)
            labels = label.to(device=device, dtype=torch.int64)
            preds = model(img)
            _, pred_class = torch.max(preds.data, 1)
            _acts = {_ix: {} for _ix in range(img.shape[0])}
            for k, v in activations.items():
                w = np.array(v.cpu().detach())
                for _ix in range(w.shape[0]):
                    _acts[_ix][k] = w[_ix].flatten()

            for _ix, (p, lbl, pred) in enumerate(zip(path, label, pred_class)):
                log.append({
                    'path': p,
                    'label': lbl.item(),
                    'pred': pred.item(),
                    'activation': _acts[_ix],
                })

    return log


def cluster(log, savefig, nclusters=2):
    from src.visualization import k_means
    nl = []
    for item in log:
        ni = {**item}
        #ni['activation'] = ni['activation']['layer4']
        ni['activation'] = ni['activation']['avgpool']
        nl.append(ni)
    acts = np.stack([item['activation'] for item in nl], axis=0)
    print(acts.shape)
    trans = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
    ])
    # l_images = [trans(Image.open(item['path']).convert("RGB")) for item in nl]
    l_images = [preprocess(item['path'], item['face'], trans) for item in nl]

    from src.visualization import scatterplot_images
    out = k_means(acts, n_clusters=nclusters)
    k_lbl, centers = out['labels'], out['centers']
    lbl2x0 = {lbl: 64*1.5*ix+64 for lbl, ix in enumerate(np.unique(k_lbl))}
    cnt_lbl = {lbl: 0 for lbl in np.unique(k_lbl)}
    x, y = [], []
    for img, lbl in zip(l_images, k_lbl):
        x.append(lbl2x0[lbl])
        y.append(cnt_lbl[lbl]*64*1.5+32)
        cnt_lbl[lbl] += 1
    data = np.stack([x, y], axis=1)
    scatterplot_images(data, l_images, savefig)

import os
import datetime
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import KLDivLoss, L1Loss, NLLLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.resnet import resnet18
from PIL import Image, ImageDraw, ImageFont
from src.utils import load_config
from src.datasets import unpickle_imdb, ImdbDataset
from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD
from src import LOG_DIR, FONT_ROOT


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

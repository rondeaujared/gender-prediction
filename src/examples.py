
def gender_estimation(weights=None):
    import os
    import datetime
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
    from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD
    load_config()
    device = torch.device('cuda')
    imdb_root = os.environ['IMDB_ROOT']
    df = unpickle_imdb(f"{imdb_root}/imdb.pickle")
    savedir = f"{os.environ['LOG_DIR']}"
    trans = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = ImdbDataset(root=imdb_root, df=df[:100000], transform=trans)
    tr, val = random_split(ds, [len(ds)-len(ds)//10, len(ds)//10])
    loss_fn = CrossEntropyLoss()
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if weights:
        model.load_state_dict(torch.load(weights))
    model.to(device)
    optim = Adam(model.parameters(), lr=3e-4)
    tr_dl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
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

    def _save_weights(suffix=''):
        time = datetime.datetime.now()
        s = f"{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}_{suffix}.pth"
        fname = f"{savedir}/{s}"
        print(f"Saving to: {fname}")
        torch.save(model.state_dict(), fname)

    epoch = 0
    for i in range(10):
        tr_log[epoch] = []
        _epoch(True)
        print_log_epoch(epoch, tr_log)
        epoch += 1
    val_log[epoch] = []
    _epoch(False)
    print_log_epoch(epoch, val_log, pretext='VAL::')
    _save_weights()


def gender_inference(root, weights):
    import os
    import torch
    from torch import nn
    from torchvision import transforms
    from torchvision.models.resnet import resnet18
    from PIL import Image, ImageDraw, ImageFont
    from src import LOG_DIR, FONT_ROOT
    from src.face_detection import (predict_faces, preds_to_txt, txt_to_preds, draw_faces, extract_faces)
    from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD

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

    def gender_preds_to_txt(file_preds, log_path):
        with open(log_path, 'w') as f:
            out = [f"{path} {pred['x1']} {pred['y1']} {pred['width']} {pred['height']} {pred['gender']}\n"
                   for path, pred in file_preds]
            f.writelines(out)

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

    def predict_gender(path_face, weights):
        device = torch.device('cuda')
        trans = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        if weights:
            model.load_state_dict(torch.load(weights))
        model.to(device)
        model.eval()
        l_preds = []
        with torch.no_grad():
            for path, face in path_face:
                img = preprocess(path, face, trans).unsqueeze(0).to(device=device)
                preds = model(img)
                _, pred_class = torch.max(preds.data, 1)
                l_preds.append((path, {**face, 'gender': pred_class.item()}))
        return l_preds

    # face_preds = draw_faces(l_faces, f'tmp/drawn_faces')
    path = extract_faces(root)
    l_faces = txt_to_preds(path)
    path_face = [
        (path, dict(face)) for path, flist in l_faces for face in flist
    ]
    print(path_face)
    l_genders = predict_gender(path_face, weights)
    print(l_genders)
    log = os.path.join(f"{LOG_DIR}", "images_gender_preds.txt")
    gender_preds_to_txt(l_genders, log)
    l_genders = gender_txt_to_preds(log)
    draw_genders(l_genders, 'tmp/drawn_genders')


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
    #_weights = '/mnt/fastdata/cnn-training-logs/6_27_13_45_49_.pth'
    _weights = '/mnt/fastdata/cnn-training-logs/6_28_12_25_7_.pth'
    # _path = '/mnt/fastdata/anno-ai/gender/streetview'
    _path = '/mnt/fastdata/challenging-binary-age/adult/adult-hat'
    # gender_estimation()
    gender_inference(_path, _weights)

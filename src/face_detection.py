import torch
import json
import os

from src.convnets.s3fd import (s3fd, nms, detect_fast, preprocess_face)
from src.utils import load_config
load_config()

FACE_MODEL_WEIGHT_PATH = os.environ.get('FACE_MODEL_WEIGHT_PATH')
MIN_SCORE = float(os.environ.get('MIN_SCORE'))
MIN_FACE_SIZE = int(os.environ.get('MIN_FACE_SIZE'))
DATA_ROOT = os.environ.get('DATAROOT')
DEVICE = torch.device('cuda')


def bboxlist_to_faces(bboxlist):
    faces = []
    if bboxlist is not None:
        for ix in range(bboxlist.shape[1]):
            lst = []
            curr = bboxlist[:, ix, :]
            keep = nms(curr, 0.30)
            curr = curr[keep, :]
            for b in curr:
                x1, y1, x2, y2, s = b
                x1, x2 = int(max(0, x1)), int(max(0, x2))
                y1, y2 = int(max(0, y1)), int(max(0, y2))
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if s < MIN_SCORE or width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                    pass
                elif width <= 0 or height <= 0 or x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or x1 > x2 or y1 > y2:
                    pass
                    print(f"problem: {width} {height} {x1} {x2} {y1} {y2}")
                else:
                    #faces.append({
                    lst.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'width': int(width),
                        'height': int(height),
                    })
            faces.append(lst)
    return faces


def predict_face(model, paths):
    l_preds = []
    with torch.no_grad():
        for path in paths:
            img = torch.Tensor(preprocess_face(path)).unsqueeze(0).to(device=DEVICE)
            preds = model(img)
            bboxlist = detect_fast(preds)
            faces = bboxlist_to_faces(bboxlist)[0]
            l_preds.append((path, faces))
    return l_preds


def predict_faces(paths):
    """
    :param paths: list of filepaths to images OR path to text file containing line seperated paths
    :return: [*(str: path, [*{'x1', 'y1', 'width', 'height'}])]
    """
    if isinstance(paths, str):
        paths = [path.strip('\n') for path in open(paths, 'r').readlines()]
    face_net = s3fd()
    face_net.load_weights(FACE_MODEL_WEIGHT_PATH)
    face_net.to(device=DEVICE)
    face_net.eval()
    l_preds = predict_face(face_net, paths)
    return l_preds


def preds_to_txt(file_preds, log_path):
    """
    :param log_path:
    :param file_preds: [*(str: path, [*{'x1', 'y1', 'width', 'height'}])]
    :return:
    """
    with open(log_path, 'w') as f:
        out = f""
        for path, preds in file_preds:
            out += f"{path}\n{len(preds)}\n"
            for pred in preds:
                out += f"{pred['x1']} {pred['y1']} {pred['width']} {pred['height']}\n"
        f.write(out)


def txt_to_preds(log_path):
    def n(_f):
        return _f.readline().strip('\n')
    file_preds = []
    with open(log_path, 'r') as f:
        while f.readable():
            path = n(f)
            if not path:
                break
            nfaces = int(n(f))
            faces = []
            for _ in range(nfaces):
                x1, y1, width, height = n(f).split(' ')
                face = {
                    'x1': int(x1),
                    'y1': int(y1),
                    'width': int(width),
                    'height': int(height),
                }
                faces.append(face)
            file_preds.append((path, faces))
    return file_preds

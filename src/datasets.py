import numpy as np
import pandas as pd
import scipy.io
import os
from datetime import timedelta
from torch.utils.data import Dataset
from PIL import Image
from src import APPA_ROOT

###################
# BEGIN IMDB-Wiki #
###################
class ImdbDataset(Dataset):
    def __init__(self, root, df, transform, target_transform=None, include_path=False):
        assert os.path.isdir(root)
        self.root = root
        self.trans = transform
        self.data = []
        self.include_path = include_path
        for row in df.itertuples():
            path = f"{self.root}/{row[3]}"
            gender = row[4]
            try:
                gender = int(gender)
            except ValueError:
                continue
            self.data.append((path, gender))

    def __getitem__(self, index):
        path, label = self.data[index]
        img = self.trans(Image.open(path).convert("RGB"))
        if self.include_path:
            return img, int(label), path
        else:
            return img, int(label)

    def __len__(self):
        return len(self.data)


def unpickle_imdb(path):
    return pd.read_pickle(path)


def build_imdb(fname, n=None, save=None, min_age=0, max_age=100):
    mat = scipy.io.loadmat(fname)
    data = mat['imdb']
    ndata = {n: data[n][0, 0][0] for n in data.dtype.names}

    celeb_names = ndata['celeb_names']
    cn = np.apply_along_axis(lambda x: x[0], 1, celeb_names.reshape(-1, 1))
    cols = {k: ndata[k] for k in ndata.keys() if k != 'celeb_names'}

    df = pd.DataFrame(cols)
    df['celeb_names'] = cn[df['celeb_id']-1]
    origin = np.datetime64('0000-01-01', 'D') - np.timedelta64(1, 'D')
    date = np.array(df['dob']) * np.timedelta64(1, 'D') + origin
    df['bday'] = pd.to_datetime(date, errors='coerce')
    df['date_taken'] = pd.to_datetime(df['photo_taken'], yearfirst=True, format='%Y')
    df['age'] = (df['date_taken'] - df['bday']) / timedelta(days=365)

    # Get rows with exactly 1 face
    print(f"N images orig {len(df)}")
    df = df[df['second_face_score'].isna()]
    print(f"N images with 1 face: {len(df)}")
    df = df[df['face_score'] > 0]
    df = df[(df.age >= min_age) & (df.age <= max_age)]
    df = df.drop(columns=['celeb_id', 'dob', 'name', 'second_face_score'])
    df['full_path'] = df.full_path.apply(lambda x: x[0]).astype(str)

    # In x1,y1,x2,y2
    df['face_location'] = df.face_location.apply(lambda x: x[0].astype(int))
    df['dx'] = df['face_location'].apply(lambda x: x[2]-x[0])
    df['dy'] = df['face_location'].apply(lambda x: x[3]-x[1])
    df = df[df['face_score'] > 2]
    print(f"N images with 1 face and high score: {len(df)}")
    print(df['face_score'].describe())
    ages = df['age'].apply(lambda x: round(x)).astype(int)
    count = {i: len(ages[ages == i]) for i in range(min_age, max_age+1)}
    weights = {i: 1/(max(count.get(i, 1), 1) / len(ages)) for i in range(min_age, max_age+1)}
    df['weight'] = ages.apply(lambda x: weights[x])

    if n is None:
        df = df[['face_location', 'face_score',
                   'full_path', 'gender', 'photo_taken', 'celeb_names', 'bday',
                   'date_taken', 'age', 'dx', 'dy', 'weight']]
    else:
        df = df.sample(n=n, weights=df.weight)[['face_location', 'face_score',
                                              'full_path', 'gender', 'photo_taken', 'celeb_names', 'bday',
                                              'date_taken', 'age', 'dx', 'dy', 'weight']]
    if save:
        df.to_pickle(save)
    return df
###################
#  END IMDB-Wiki  #
###################


####################
#  BEGIN VGGFace2  #
####################
class VGGFaceDataset(Dataset):
    def __init__(self, root, trans, test=False, nidents=None, include_path=False):
        """
        Expects data to be stored as follows:
        root/[test_list.txt, train_list.txt, vggface2_test, vggface2_train]
        root/vggface2_test/test/[nXXXXXX, ...]/[YYYY_YY.jpg, ...]
        root/vggface2_train/train/[nXXXXXX, ...]/[YYYY_YY.jpg, ...]
        :param root:  Absolute path to base folder containing vggface2
        :param trans: PyTorch transformer
        :param test:  [True/False] if we should use test split
        :param include_path:  [True/False] to include abs path to image in label
        """
        split = 'test' if test is True else 'train'
        with open(os.path.join(root, f'{split}_list.txt'), 'r') as f:
            lines = f.read().splitlines()
        data = []
        identity = {line.split('/')[0]: [] for line in lines}

        if nidents:
            identity = {ident: imgs for ident, imgs in list(identity.items())[:nidents]}

        for line in lines:
            iden, img = line.split('/')
            if nidents and iden not in identity:
                continue
            data.append(os.path.join(root, f'vggface2_{split}/{split}/{line}'))
            identity[iden].append(img)

        self.root = root
        self.trans = trans
        self.include_path = include_path
        self.identity = identity
        self.data = data

    def getiden(self, iden):
        return self.identity.get(iden, None)

    def __getitem__(self, idx):
        path = self.data[idx]
        img = self.trans(Image.open(path).convert("RGB"))
        out = [img]
        if self.include_path: out.append(path)
        #return tuple(out)
        return (img, path)

    def __len__(self):
        return len(self.data)

####################
#   END VGGFace2   #
####################


###################
# BEGIN APPA-REAL #
###################
class AppaRealDataset(Dataset):
    def __init__(self, trans, split, target_trans=None, faceonly=True):
        assert split in ['train', 'val', 'test']
        self.root = APPA_ROOT
        self.split = split
        self.faceonly = faceonly
        self.trans = trans
        if target_trans:
            self.target_trans = target_trans
        else:
            self.target_trans = self.gender_target
        with open(os.path.join(self.root, f'gt_avg_{self.split}.csv')) as f:
            lines = f.read().splitlines()
        header = lines.pop(0).split(',')
        files = {}
        for line in lines:
            file_name, num_ratings, app_avg, app_std, real_age = line.split(',')
            files[file_name] = {
                'num_ratings': num_ratings,
                'app_avg': app_avg,
                'app_std': app_std,
                'real_age': real_age,
            }
        with open(os.path.join(self.root, f'allcategories_{self.split}.csv')) as f:
            lines = f.read().splitlines()
        header = lines.pop(0).split(',')
        for line in lines:
            file_name, gender, race, makeup, time, happiness = line.split(',')
            files[file_name] = {
                **files[file_name],
                'gender': gender,
                'race': race,
                'makeup': makeup,
                'time': time,
                'happiness': happiness,
            }
        self.files = files
        self.data = [k for k in files.keys()]

    def __getitem__(self, idx):
        fname = self.data[idx]
        info = self.files[fname]
        img_path = os.path.join(self.root, f'{self.split}/{fname}')
        if self.faceonly: img_path += '_face.jpg'
        img = Image.open(img_path).convert("RGB")
        img = self.trans(img)
        label = self.target_trans(info)
        return img, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def gender_target(info):
        return 1 if info['gender'] == 'male' else 0

###################
#  END APPA-REAL  #
###################

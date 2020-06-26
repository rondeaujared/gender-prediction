import numpy as np
import pandas as pd
import scipy.io
from datetime import timedelta


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
    df = df[df['second_face_score'].isna()]
    df = df[df['face_score'] > 0]
    df = df[(df.age >= min_age) & (df.age <= max_age)]
    df = df.drop(columns=['celeb_id', 'dob', 'name', 'second_face_score'])
    df['full_path'] = df.full_path.apply(lambda x: x[0]).astype(str)

    # In x1,y1,x2,y2
    df['face_location'] = df.face_location.apply(lambda x: x[0].astype(int))
    df['dx'] = df['face_location'].apply(lambda x: x[2]-x[0])
    df['dy'] = df['face_location'].apply(lambda x: x[3]-x[1])
    df = df[df['face_score'] > 2]
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

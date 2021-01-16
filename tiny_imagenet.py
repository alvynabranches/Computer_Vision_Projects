import os
import json
import time
import threaded
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.image import imread

# import torch
# import torch.nn as nn
# import torch.functional as F
# import torchvision

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
# from tensorflow.keras.optimizers import Adam

base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, 'data/tiny-imagenet-200/')
train_data_path, test_data_path, val_data_path = data_path + 'train/', data_path + 'test/', data_path + 'val/'
SHUTDOWN = True

def get_paths(path, csv_file_path='./data/data.csv'):
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame(columns=['filename', 'label_1', 'label_2', 'label_3', 'label_4'])
        for root, dirs, files in os.walk(path):
            if root != path:
                for f in files:
                    if str(f).endswith('.txt'):
                        df = df.append(pd.read_csv(os.path.join(root, f), sep='\t', names=['filename', 'label_1', 'label_2', 'label_3', 'label_4']))
        df.to_csv(csv_file_path, index=False)
    total_imgs = df.shape[0]
    img_paths = []
    for i, (root, dirs, files) in enumerate(os.walk(path)):
        if root != path:
            for f in files:
                if str(f).endswith('.JPEG'):
                    img_paths.append(str(os.path.join(root, f)))
    return df, img_paths

def get_data(img_pths:list, df, json_file_path='./data/data.json'):
    @threaded.Threaded()
    def save(X, y, json_file_path):
        with open(json_file_path, 'w') as f:
            json.dump(dict(X=X[1:].tolist(), y=y[1:].tolist()), f, indent=4)
    X = np.zeros((1,64,64,3))
    y = np.zeros((1,4))
    error = 0
    done = 0
    total = len(img_pths)
    for f in img_pths:
        try:
            done += 1
            X = np.append(X, imread(f)[np.newaxis,...], axis=0)
            y = np.append(y, np.array(df[df['filename']==str(f).split('\\')[-1]][['label_1', 'label_2', 'label_3', 'label_4']].values.tolist()), axis=0)
        except Exception as e:
            print(f'Exception: {f}', end='\r')
            error += 1
        print(f'\r{X.shape=}  {y.shape=}  {error=}  {X.shape[0]==y.shape[0]}  {done/total*100:.2f}%', end='\r')
    save(X, y, json_file_path).start()
    return X, y

def generate_data(itrs, start=0):
    btcs = int(100_000 / itrs)
    df, imgs = get_paths(train_data_path)
    for i in range(start, itrs):
        s = time.perf_counter()
        _, _ = get_data(imgs[i*btcs:(i+1)*btcs], df, f'./data/{btcs}_{i+1}.json')
        del _
        e = time.perf_counter()
        print()
        print(f'Time Taken: {(e-s):.2f} seconds')



generate_data()


os.system('shutdown /s /t 15')
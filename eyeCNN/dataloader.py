import matplotlib.pyplot as plt
import numpy as np
import os

from six.moves import cPickle as pickle
import cv2

dirs = [r'F:\aria2\dataset\dataset_B_Eye_Images\openLeftEyes', r'F:\aria2\dataset\dataset_B_Eye_Images\openRightEyes']
dirs2 = [r'F:\aria2\dataset\dataset_B_Eye_Images\closedLeftEyes', r'F:\aria2\dataset\dataset_B_Eye_Images\closedRightEyes']


def generate_dataset():
    # 修改数据集大小以包含三个通道
    dataset = np.ndarray([1231 * 2, 24, 24, 3], dtype='float32')
    i = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):
                im = cv2.imread(dir + '/' + filename, cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, (24, 24))
                im = np.array(im, dtype='float32') / 255.0
                # 复制灰度数据到三个颜色通道
                im = np.stack([im]*3, axis=-1)
                dataset[i, :, :, :] = im
                i += 1
    labels = np.ones([len(dataset), 1], dtype=int)
    return dataset, labels

def generate_dataset_closed():
    dataset = np.ndarray([1192 * 2, 24, 24, 3], dtype='float32')
    i = 0
    for dir in dirs2:
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):
                im = cv2.imread(dir + '/' + filename, cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, (24, 24))
                im = np.array(im, dtype='float32') / 255.0
                # 复制灰度数据到三个颜色通道
                im = np.stack([im]*3, axis=-1)
                dataset[i, :, :, :] = im
                i += 1
    labels = np.zeros([len(dataset), 1], dtype=int)
    return dataset, labels


dataset_open, labels_open = generate_dataset()
dataset_closed, labels_closed = generate_dataset_closed()
print("done")

split = int(len(dataset_closed) * 0.8)
train_dataset_closed = dataset_closed[:split]
train_labels_closed = labels_closed[:split]
test_dataset_closed = dataset_closed[split:]
test_labels_closed = labels_closed[split:]

pickle_file = 'closed_eyes.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_closed,
        'train_labels': train_labels_closed,
        'test_dataset': test_dataset_closed,
        'test_labels': test_labels_closed,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

split = int(len(dataset_open) * 0.8)
train_dataset_open = dataset_open[:split]
train_labels_open = labels_open[:split]
test_dataset_open = dataset_open[split:]
test_labels_open = labels_open[split:]

pickle_file = 'open_eyes.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_open,
        'train_labels': train_labels_open,
        'test_dataset': test_dataset_open,
        'test_labels': test_labels_open,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

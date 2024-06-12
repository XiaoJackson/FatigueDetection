import numpy as np
import os
import cv2

from six.moves import cPickle as pickle

dirs = [r'F:\aria2\dataset\archive\yawn', r'F:\aria2\dataset\archive\noyawn']
countYawn = 2528
countNormal = 2591

def generate_dataset():
    maxX = 60
    maxY = 60
    dataset = np.ndarray([countYawn + countNormal, maxY, maxX, 1], dtype='float32')
    i = 0
    j = 0
    pos = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):
                im = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                im = cv2.resize(im, (maxX, maxY))
                im = np.expand_dims(im, axis=-1)  # 扩展最后一个维度
                im = im / 255.0  # 归一化
                dataset[i, :, :, :] = im[:, :, :]
                i += 1
        if pos == 0:
            labels = np.ones([i, 1], dtype=int)
            j = i
            pos += 1
        else:
            labels = np.concatenate((labels, np.zeros([i - j, 1], dtype=int)))
    return dataset, labels

dataset, labels = generate_dataset()
print("Total = ", len(dataset))

totalCount = countYawn + countNormal
split = int(countYawn * 0.8)
splitEnd = countYawn
split2 = countYawn + int(countNormal * 0.8)

train_dataset = dataset[:split]
train_labels = np.ones([split, 1], dtype=int)
test_dataset = dataset[split:splitEnd]
test_labels = np.ones([splitEnd - split, 1], dtype=int)

train_dataset = np.concatenate((train_dataset, dataset[splitEnd:split2]))
train_labels = np.concatenate((train_labels, np.zeros([split2 - splitEnd, 1], dtype=int)))
test_dataset = np.concatenate((test_dataset, dataset[split2:]))
test_labels = np.concatenate((test_labels, np.zeros([totalCount - split2, 1], dtype=int)))

pickle_file = 'yawn_mouths.pickle'

try:
    with open(pickle_file, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

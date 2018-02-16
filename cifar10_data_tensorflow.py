#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# download and read CIFAR-10
import os
import numpy as np
import pickle
import sys
import tarfile
from six.moves import urllib
import numpy as np


def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(data_dir + '/cifar-10-batches-py'):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)



def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

def load_cifar_data(data_dir):
    maybe_download_and_extract(data_dir)
    xs = []
    ys = []
    for j in range(5):
        d = unpickle(data_dir + '/cifar-10-batches-py/data_batch_' + str(j + 1))
        # d = unpickle(data_dir + '/cifar-10-batches-py/data_batch_'+'j+1')
        x = d['data']
        y = d['labels']
        xs.append(x)
        ys.append(y)

    d = unpickle(data_dir + '/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])
    # 默认情况下，axis=0可以不写按行连接，【10000*6，3072】,除255由unit8变成float型进行计算
    x = np.concatenate(xs) / np.float32(255)
    # 【10000*6， 1】
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    x = x.reshape(-1, 32 * 32 * 3)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000], axis=0)
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000, :]
    Y_train = y[0:50000]#label
    Y_train_onehot=dense_to_one_hot(Y_train)
    X_test = x[50000:, :]
    Y_test = y[50000:]#label
    Y_test_onehot=dense_to_one_hot(Y_test)

    return pixel_mean, dict(
        X_train=X_train,#[50000,3072]
        Y_train=Y_train.astype('int32'),#[50000,1]
        Y_train_onehot=Y_train_onehot,#[50000,10]
        X_test=X_test,
        Y_test=Y_test.astype('int32'),
        Y_test_onehot=Y_test_onehot,
    )

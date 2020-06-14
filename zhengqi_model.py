# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15 22:07
@Author  : ds
@File    : zhengqi_model.py
Have a nice day!
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DATA_PATH = './data/zhengqi_train.txt'
TEST_DATA_PATH = './data/zhengqi_test.txt'


def LoadData(dataPath):
    # load data from txt file
    data = np.loadtxt(dataPath, dtype='float', delimiter='\t', skiprows=1)
    data_features = data[:, :38]#提取前38列为特征数据
    data_labels = data[:, -1]#提取最后一列为标签数据
    return (data_features, data_labels)

def NormalizeData(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    return ((train_data - mean)/std, (test_data - mean)/std)

def SplitData(train_fetures, train_labels):
    # split data into train and valid data
    train_index = np.random.choice(len(train_fetures), round(len(train_fetures) * 0.8), replace=False)
    valid_index = np.array(list(set(range(len(train_fetures))) - set(train_index)))
    train_data_features = train_fetures[train_index]
    valid_data_features = train_fetures[valid_index]
    train_data_labels = train_labels[train_index]
    valid_data_labels = train_labels[valid_index]
    return (train_data_features, train_data_labels), (valid_data_features, valid_data_labels)

def CreataModel():
    model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=[38]),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

def main():
    (train_fetures, train_labels) = LoadData(TRAIN_DATA_PATH)
    (test_fetures, test_labels) = LoadData(TEST_DATA_PATH)
    (train_fetures, test_fetures) = NormalizeData(train_fetures, test_fetures)
    (train_data_features, train_data_labels), (valid_data_features, valid_data_labels) = SplitData(train_fetures, train_labels)

    model = CreataModel()

    history = model.fit(train_data_features,
                        train_data_labels,
                        epochs=32,
                        batch_size=64,
                        validation_data=(valid_data_features, valid_data_labels),
                        verbose=1)

    test_predictions = model.predict(test_fetures)
    np.savetxt('./submit/njit_77.txt', test_predictions)

if __name__ == '__main__':
    main()


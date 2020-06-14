# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15 22:07
@Author  : ds
@File    : zhengqi_model.py
Have a nice day!
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DATA_PATH = './data/zhengqi_train.txt'
TEST_DATA_PATH = './data/zhengqi_test.txt'


def LoadData(dataPath):
    # 加载txt文本
    data = np.loadtxt(dataPath, dtype='float', delimiter='\t', skiprows=1)
    data_features = data[:, :38]#提取前38列为特征数据
    data_labels = data[:, -1]#提取最后一列为标签数据
    return (data_features, data_labels)

def NormalizeData(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    return ((train_data - mean)/std, (test_data - mean)/std)

#构造模型
def CreataModel():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[38]),
        keras.layers.Dropout(0.5),
        layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
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

    model = CreataModel()
    history = model.fit(train_fetures,
                        train_labels,
                        epochs=20,
                        batch_size=64,
                        validation_split=0.2,
                        verbose=2)
    test_predictions = model.predict(test_fetures)
    np.savetxt('./submit/njit_77.txt', test_predictions)

if __name__ == '__main__':
    main()


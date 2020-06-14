# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15 22:07
@Author  : ds
@File    : zhengqi_model.py
Have a nice day!
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import pca

TRAIN_DATA_PATH = './data/zhengqi_train.txt'
TEST_DATA_PATH = './data/zhengqi_test.txt'


def LoadData():
    # load data from txt file
    train_data = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    test_data = pd.read_csv(TEST_DATA_PATH, sep='\t')
    return (train_data, test_data)

def NormalizeData(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    return ((train_data - mean)/std, (test_data - mean)/std)

def main():

    if True:
        (train_data, test_data) = LoadData()

        train_data.drop(["V27"], axis=1, inplace=True)
        test_data.drop(["V27"], axis=1, inplace=True)

        train_data_x = train_data.drop(['target'], axis=1)
        train_data_y = train_data['target']

        """
                           本地测试(mse)         线上score        时间 
            False True  0.10886303677029652      0.1631     2019-02-01 11:59:00
            True  True  0.11513819403279449      0.1840     2019-02-01 09:13:00
            True  False 0.09077965719786232      0.1267     2019-01-31 21:55:00
            False False 0.08781269343785499      0.1273     2019-01-26 18:46:00
        """
        if True:
            (train_data_x, test_data) = NormalizeData(train_data_x, test_data)
        if False:
            pca1 = pca.PCA(n_components=0.95)
            pca1.fit(train_data_x)
            train_data_x = pca1.transform(train_data_x)
            test_data = pca1.transform(test_data)

        X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=40)
        params = {'learning_rate': 0.03,
                  'loss': 'huber',
                  'max_depth': 14,
                  'max_features': 'sqrt',
                  'min_samples_leaf': 10,
                  'min_samples_split': 40,
                  'n_estimators': 300,
                  'random_state': 10,
                  'subsample': 0.8}
        clf = GradientBoostingRegressor(**params)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(mean_squared_error(Y_test, Y_pred))

        if True:
            test_data_y = clf.predict(test_data)
            res_pd = pd.DataFrame(test_data_y, columns=['target'])
            res_pd.to_csv("./submit/njit_77.txt", index=False, header=False)

if __name__ == '__main__':
    main()

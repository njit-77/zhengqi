# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15 22:07
@Author  : ds
@File    : zhengqi_model.py
Have a nice day!
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import pca
from sklearn.ensemble import GradientBoostingRegressor

TRAIN_DATA_PATH = './data/zhengqi_train.txt'
TEST_DATA_PATH = './data/zhengqi_test.txt'


def LoadData():
    # load data from txt file
    train_data = pd.read_table(TRAIN_DATA_PATH)
    test_data = pd.read_table(TEST_DATA_PATH)
    train_data_x = train_data.values[:, 0:-1]
    train_data_y = train_data.values[:, -1]
    return (train_data_x, train_data_y, test_data)

def main():
    (train_data_x, train_data_y, test_data) = LoadData()

    pca1 = pca.PCA(n_components=0.95)
    pca1.fit(train_data_x)
    train_data_x = pca1.transform(train_data_x)
    test_data = pca1.transform(test_data)

    X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=40)
    myGBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                      learning_rate=0.03, loss='huber', max_depth=14,
                                      max_features='sqrt', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                      min_samples_leaf=10, min_samples_split=40,
                                      min_weight_fraction_leaf=0.0, n_estimators=300,
                                      presort='auto', random_state=10, subsample=0.8, verbose=0,
                                      warm_start=False)
    myGBR.fit(X_train, Y_train)
    Y_pred = myGBR.predict(X_test)
    print(mean_squared_error(Y_test, Y_pred))

    '''结果预测'''
    test_data_y = myGBR.predict(test_data)

    res_pd = pd.DataFrame(test_data_y, columns=['target'])
    res_pd.to_csv("./submit/njit_77.txt", index=False, header=False)
    print("over")

if __name__ == '__main__':
    main()


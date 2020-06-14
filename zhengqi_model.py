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
    train_data = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    test_data = pd.read_csv(TEST_DATA_PATH, sep='\t')
    return (train_data, test_data)

def main():
    (train_data, test_data) = LoadData()

    """
                   本地测试(mse)         线上score        时间 
    False True  0.10895188423219443      0.1682     2019.1.19 20:00
    True  True  0.11475080628985444      0.1585     2019.1.20 20:00
    True  False 0.09424565768574228      0.1373     2019.1.21 20:00
    False False 0.08918637637655073
    """
    if True:
        train_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
        test_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

    train_data_x = train_data.drop(['target'], axis=1)
    train_data_y = train_data['target']

    if False:
        pca1 = pca.PCA(n_components=0.95)
        pca1.fit(train_data_x)
        train_data_x = pca1.transform(train_data_x)
        test_data = pca1.transform(test_data)

    X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=40)
    myGBR = GradientBoostingRegressor(learning_rate=0.03,
                                      loss='huber',
                                      max_depth=14,
                                      max_features='sqrt',
                                      min_samples_leaf=10,
                                      min_samples_split=40,
                                      n_estimators=300,
                                      random_state=10,
                                      subsample=0.8)
    myGBR.fit(X_train, Y_train)
    Y_pred = myGBR.predict(X_test)
    print(mean_squared_error(Y_test, Y_pred))

    '''结果预测'''
    test_data_y = myGBR.predict(test_data)

    res_pd = pd.DataFrame(test_data_y, columns=['target'])
    res_pd.to_csv("./submit/njit_77.txt", index=False, header=False)

if __name__ == '__main__':
    main()


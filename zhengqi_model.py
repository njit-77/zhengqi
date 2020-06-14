# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/15 22:07
@Author  : ds
@File    : zhengqi_model.py
Have a nice day!
"""


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd  # 导入pandas库
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

    train_data.drop(['V27'], axis=1, inplace=True)
    test_data.drop(['V27'], axis=1, inplace=True)

    train_data_x = train_data.drop(['target'], axis=1)
    train_data_y = train_data['target']

    if True:
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

    '''结果预测'''
    if True:
        test_data_y = clf.predict(test_data)
        res_pd = pd.DataFrame(test_data_y, columns=['target'])
        res_pd.to_csv("./submit/njit_77.txt", index=False, header=False)

if __name__ == '__main__':
    main()

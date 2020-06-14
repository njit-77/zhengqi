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

    if False:
        """
        Drop V0, MES = 0.09276847269977244
        Drop V1, MES = 0.09615506562076584
        Drop V2, MES = 0.09205514264630522
        Drop V3, MES = 0.09918668851070986
        Drop V4, MES = 0.09250390382500329
        Drop V5, MES = 0.09027781546961815
        Drop V6, MES = 0.08991929667285999
        Drop V7, MES = 0.09205849450412687
        Drop V8, MES = 0.0910798844411952
        Drop V9, MES = 0.09045514527508201
        Drop V10, MES = 0.09549311100439208
        Drop V11, MES = 0.09258864213179133
        Drop V12, MES = 0.09019494163018743
        Drop V13, MES = 0.09003051760686039
        Drop V14, MES = 0.09169395549410853
        Drop V15, MES = 0.09237638957685337
        Drop V16, MES = 0.09286382760308219
        Drop V17, MES = 0.09041601517586591
        Drop V18, MES = 0.09028528279505393
        Drop V19, MES = 0.09213564275596815
        Drop V20, MES = 0.0929645280782623
        Drop V21, MES = 0.09106172941420454
        Drop V22, MES = 0.09155541609466313
        Drop V23, MES = 0.09131439239385983
        Drop V24, MES = 0.09154908893714261
        Drop V25, MES = 0.0908609008176204
        Drop V26, MES = 0.09147964137176223
        Drop V27, MES = 0.08781269343785499
        Drop V28, MES = 0.08885483546118565
        Drop V29, MES = 0.09018992337075467
        Drop V30, MES = 0.09322131747790884
        Drop V31, MES = 0.09032886873680467
        Drop V32, MES = 0.0919657292056093
        Drop V33, MES = 0.09242582557448874
        Drop V34, MES = 0.0923314534528684
        Drop V35, MES = 0.09312726505053395
        Drop V36, MES = 0.09310052060903348
        Drop V37, MES = 0.09157560338800454
        """
        for col_name in range(0, 38):
            (train_data, test_data) = LoadData()
            train_data.drop(["V{0}".format(col_name)], axis=1, inplace=True)
            test_data.drop(["V{0}".format(col_name)], axis=1, inplace=True)

            train_data_x = train_data.drop(['target'], axis=1)
            train_data_y = train_data['target']

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
            print("Drop V{0}, MES = {1}".format(col_name, mean_squared_error(Y_test, Y_pred)))

    if True:
        (train_data, test_data) = LoadData()

        train_data.drop(["V28"], axis=1, inplace=True)
        test_data.drop(["V28"], axis=1, inplace=True)

        train_data_x = train_data.drop(['target'], axis=1)
        train_data_y = train_data['target']

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

from time import process_time
import pandas as pd
import matplotlib.pyplot as plt

from nerual_network_regression import *
from nerual_network_classification import *
from other_func import *


def main_regression():
    data = pd.read_csv('./data/regression_data.txt', index_col=None, header=None)
    X = data.iloc[:, 0].values.astype(float)
    y = data.iloc[:, 1].values.astype(float)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.75, random_state=1)
    start = process_time()
    ann = BPNeuralNetworkRegression(solver='mbgd', learning_rate=0.4, max_iter=10000,
                                    batch_size=32,
                                    is_print=True,
                                    normalize=True,
                                    activation='ReLU',
                                    init_method='Xavier',
                                    goal=1e-4, tol=1e-12,
                                    alpha=1e-4, momentum=0.9,
                                    clip_gradient=5, how_clip='normal',
                                    hidden_layer_sizes=15,
                                    )
    ann.fit(X_train, y_train)
    y_predict = ann.predict(X)
    plt.plot(X, y, color='r')
    plt.scatter(X, y_predict, color='b')
    plt.show()
    print('\n相关系数: ', calc_corr(y_predict, y))
    print("Time used:", process_time() - start)


def main_classification():
    class1 = pd.read_csv('./data/classification_data1.txt', index_col=None, header=None)
    class2 = pd.read_csv('./data/classification_data2.txt', index_col=None, header=None)
    SA_value = 0.2
    SB_value = 0.8
    x1 = class1.iloc[:, 0].values.astype(float)
    y1 = class1.iloc[:, 1].values.astype(float)
    x2 = class2.iloc[:, 0].values.astype(float)
    y2 = class2.iloc[:, 1].values.astype(float)
    X = [np.array([x, y]) for x, y in zip(x1, y1)]
    X.extend([np.array([x, y]) for x, y in zip(x2, y2)])
    y = ([SA_value] * len(x1))
    y.extend([SB_value] * len(x2))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
    start = process_time()

    ann = BPNeuralNetworkClassification(learning_rate=0.1, max_iter=5000,
                                        solver='mbgd',
                                        batch_size=32,
                                        is_print=True,
                                        normalize=True,
                                        activation='ReLU',
                                        init_method='Xavier',
                                        goal=1e-4, tol=1e-8,
                                        alpha=1e-4, momentum=0.9,
                                        clip_gradient=5, how_clip='normal',
                                        hidden_layer_sizes=(15, 15),
                                        )
    ann.fit(X_train, y_train)
    y_predict = ann.predict(X)

    SA_num = SB_num = 0
    SA_right = SB_right = 0
    for index in range(0, len(y)):
        if y[index] == SA_value:
            SA_num += 1
            if y_predict[index] == 0:
                SA_right += 1
        elif y[index] == SB_value:
            SB_num += 1
            if y_predict[index] == 1:
                SB_right += 1
    print(y_predict)
    print(SA_right, ' ', SA_num)
    print(SB_right, ' ', SB_num)
    print('A 类分类正确率(%): ', 100 * SA_right / SA_num)
    print('B 类分类正确率(%): ', 100 * SB_right / SB_num)
    print('平均分类正确率(%): ', 50 * (SA_right / SA_num + SB_right / SB_num))
    elapsed = (process_time() - start)
    print("\nTime used:", elapsed)
    plt.scatter(x1, y1, label='SA', marker='+', color='r', )
    plt.scatter(x2, y2, label='SB', marker='o', color='', edgecolors='k')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # main_regression()
    main_classification()

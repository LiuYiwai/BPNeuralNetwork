from time import process_time
import pandas as pd
import matplotlib.pyplot as plt

from nerual_network import *
from other_func import *


def main_regression():
    data = pd.read_csv('./data/regression_data.txt', index_col=None, header=None)
    X = data.iloc[:, 0].values.astype(float)
    y = data.iloc[:, 1].values.astype(float)
    X_train, tmp1, y_train, tmp2 = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)
    tmp1, X_test, tmp2, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
    start = process_time()
    ann = BPNeuralNetworkRegression(solver='mbgd',
                                    learning_rate=0.03,
                                    max_iter=20000,
                                    is_print=True,
                                    normalized=True,
                                    goal=2.3e-3, tol=1e-12,
                                    alpha=1e-4,
                                    momentum=0.7,
                                    batch_size=32,
                                    activation='ReLU',
                                    init_method='Xavier',
                                    # hidden_layer_sizes=(30, 15),
                                    hidden_layer_sizes=30,
                                    )
    try:
        ann.fit(X_train, y_train, X_test, y_test)
    except KeyboardInterrupt:
        pass
    y_predict = ann.predict(X)
    plt.title('Fitting result')
    plt.xlim(0, 7)
    plt.plot(X, y, color='r', label='Target curve')
    plt.scatter(X, y_predict, color='b', label='Fit curve')
    plt.legend()
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
    SA_len = len(x1)
    X = [np.array([x, y]) for x, y in zip(x1, y1)]
    X.extend([np.array([x, y]) for x, y in zip(x2, y2)])
    y = ([SA_value] * len(x1))
    y.extend([SB_value] * len(x2))
    X = np.array(X)
    y = np.array(y)
    alpha = 0
    X = add_noise(X, alpha)
    plt.title('Input sample')
    plt.scatter(X[0:SA_len, 0], X[0:SA_len, 1], label='SA', marker='+', color='r', )
    plt.scatter(X[SA_len:, 0], X[SA_len:, 1], label='SB', marker='o', color='', edgecolors='k')
    plt.legend(loc='upper right')
    plt.show()

    X_train, tmp1, y_train, tmp2 = model_selection.train_test_split(X, y, test_size=0.4, random_state=1)
    tmp1, X_test, tmp2, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    start = process_time()
    ann = BPNeuralNetworkClassification(learning_rate=0.01, max_iter=5000,
                                        solver='mbgd',
                                        batch_size=32,
                                        is_print=True,
                                        normalized=True,
                                        activation='ReLU',
                                        init_method='Xavier',
                                        goal=1e-4, tol=0,
                                        alpha=1e-4, momentum=0.9,
                                        hidden_layer_sizes=(30, 20),
                                        )
    try:
        ann.fit(X_train, y_train, X_test, y_test)
    except KeyboardInterrupt:
        pass
    y_predict = ann.predict(X)

    SA_num = sum(y == SA_value)
    SB_num = len(y) - SA_num
    SA_right = sum((y == SA_value)[y_predict == 0])
    SB_right = sum((y == SB_value)[y_predict == 1])
    print('A 类分类正确率(%): ', 100 * SA_right / SA_num)
    print('B 类分类正确率(%): ', 100 * SB_right / SB_num)
    print('平均分类正确率(%): ', 50 * (SA_right / SA_num + SB_right / SB_num))
    elapsed = (process_time() - start)
    print("\nTime used:", elapsed)

    predict_SA = X[y_predict == 0]
    predict_SB = X[y_predict == 1]
    plt.title('Classification result')
    plt.scatter(predict_SA[:, 0], predict_SA[:, 1], label='SA', marker='+', color='r', )
    plt.scatter(predict_SB[:, 0], predict_SB[:, 1], label='SB', marker='o', color='', edgecolors='k')
    plt.legend(loc='upper right')
    plt.show()

    predict_SA = y_predict[0:SA_len]
    predict_SB = y_predict[SA_len:]
    plt.title('Classification result2')
    plt.scatter(range(0, SA_len), predict_SA, label='SA', marker='+', color='r', )
    plt.scatter(range(0, SA_len), predict_SB, label='SB', marker='o', color='', edgecolors='k')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main_regression()
    # main_classification()

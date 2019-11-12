from time import process_time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

from network import *


def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor


def main():
    data = pd.read_csv('out.txt', index_col=None, header=None)
    X = data.iloc[:, 0].values.astype(float)
    y = data.iloc[:, 1].values.astype(float)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
    start = process_time()
    ann = BPNeuralNetwork(learning_rate=0.001, max_iter=8000,
                          solver='bgd',
                          batch_size=32,
                          is_print=True,
                          normalize=True,
                          activation='ReLU',
                          # init_method='Xavier',
                          # init_method='MSRA',
                          goal=1e-4, tol=1e-8,
                          momentum=1e-2,
                          alpha=1e-4,
                          clip_gradient=5, how_clip='normal',
                          # hidden_layer_sizes=15,
                          hidden_layer_sizes=(15, 5),
                          # hidden_layer_sizes=(2, 3),
                          # hidden_layer_sizes=(50, 30),
                          )
    ann.fit(X_train, y_train)
    y_predict = ann.predict(X)
    plt.plot(X, y, color='r')
    plt.scatter(X, y_predict, color='b')
    plt.show()
    print('\n相关系数: ', calc_corr(y_predict, y))
    print("Time used:", process_time() - start)


if __name__ == '__main__':
    main()

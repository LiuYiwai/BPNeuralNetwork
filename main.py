import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

from network import *


def main():
    data = pd.read_csv('data.txt', index_col=None, header=None)
    X = data.iloc[:, 0].values.astype(float)
    y = data.iloc[:, 1].values.astype(float)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
    ann = BPNeuralNetwork(solver='sgd', activation='ReLU', learning_rate=0.001,
                          is_print=True,
                          goal=1e-4, tol=1e-4,
                          alpha=1e-4, max_iter=1000,
                          clip_gradient=5, how_clip='normal',  # how_clip = global or normal or none
                          hidden_layer_sizes=(50, 50))
    ann.fit(X_train, y_train)
    y_predict = ann.predict(X)

    plt.plot(X, y, color='r')
    plt.scatter(X, y_predict, color='b')
    plt.show()


if __name__ == '__main__':
    main()

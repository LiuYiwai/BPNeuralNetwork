import abc
import copy
import numpy as np
import random
import math
from sklearn import model_selection


class BPNeuralNetwork(metaclass=abc.ABCMeta):
    def __init__(self, solver='mbgd', activation='ReLU',
                 init_method='normal', normalize=False,
                 learning_rate=1e-3, momentum=None,
                 clip_gradient=5, how_clip=None, is_print=False,
                 goal=1e-4, tol=1e-4, alpha=1e-4, max_iter=500,
                 batch_size=32,
                 hidden_layer_sizes=(100,), is_classification=False):
        """
        :param solver: 权重优化的求解器
        :param activation: 激活函数
        :param learning_rate: 学习率，暂时只使用常数
        :param clip_gradient: 梯度裁剪阈值
        :param how_clip: 梯度裁剪方式 normal or global or None
        :param is_print: 是否打印每次迭代的error结果
        :param goal: 目标误差
        :param tol: 容差优化
        :param alpha: L2正则化参数
        :param max_iter: 最高迭代次数
        :param hidden_layer_sizes: 隐藏层结构
        """
        if isinstance(hidden_layer_sizes, int):
            self._layer_size = 1 + 2
            self._layer_number = (hidden_layer_sizes,)
        else:
            self._layer_size = len(hidden_layer_sizes) + 2
            self._layer_number = hidden_layer_sizes
        self._solver = solver
        self._activation = activation
        self._batch_size = batch_size
        self._init_method = init_method
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._goal = goal
        self._tol = tol
        self._alpha = alpha
        self._normalize = normalize
        self._momentum = momentum
        self._clip_gradient = clip_gradient
        self._how_clip = how_clip
        self._is_print = is_print
        self._is_classification = is_classification

    def _init_weight_layer(self, layer_size):
        if self._init_method == 'Xavier':
            scale = np.sqrt(6. / (layer_size[0] + layer_size[1]))
            return np.random.uniform(-scale, scale, size=layer_size)
        elif self._init_method == 'MSRA':
            scale = np.sqrt(2. / layer_size[0])
            return np.random.normal(0, scale, size=layer_size)
        elif self._init_method == 'normal' or True:
            scale = 1
            return np.random.normal(0, scale, size=layer_size)

    def _init_weight(self, inp, oup):
        self._input_size = inp.shape[1]
        self._output_size = oup.shape[1]
        self._layer_units = ([self._input_size] + list(self._layer_number) + [self._output_size])
        self._weight = [self._init_weight_layer(
            layer_size=(self._layer_units[i], self._layer_units[i + 1])) for i in range(0, self._layer_size - 1)]
        self._bias_weight = [self._init_weight_layer(layer_size=(1, size)) for size in self._layer_units[1:]]
        self._params = self._weight + self._bias_weight

    def _clip_by_norm(self, delta):
        t = sum([x * x for x in delta[0]])
        if t < self._clip_gradient:
            return delta
        else:
            return delta * (self._clip_gradient / t)

    def _clip_by_global_norm(self, delta):
        t = 0
        for x in delta:
            t += sum(y * y for y in x[0])
        if t < self._clip_gradient:
            return delta
        else:
            scale = self._clip_gradient / t
            return [scale * x for x in delta]

    def _get_clip_gradient(self, delta, bias_delta):
        if self._how_clip == 'normal':
            delta = [self._clip_by_norm(k * self._learning_rate / self._batch_size) for k in delta]
            bias_delta = [self._clip_by_norm(k * self._learning_rate / self._batch_size) for k in bias_delta]
        elif self._how_clip == 'global':
            delta = self._clip_by_global_norm([k * self._learning_rate / self._batch_size for k in delta])
            bias_delta = self._clip_by_global_norm([k * self._learning_rate / self._batch_size for k in bias_delta])
        elif self._how_clip is None or True:
            delta = [k * self._learning_rate / self._batch_size for k in delta]
            bias_delta = [k * self._learning_rate / self._batch_size for k in bias_delta]
        return delta, bias_delta

    @staticmethod
    def _gen_batches(n, batch_size):
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            if end > n:
                continue
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)

    def _forward_pass(self, activations):
        for i in range(0, self._layer_size - 1):
            activations[i + 1] = np.add(np.dot(activations[i], self._weight[i]), self._bias_weight[i])
            if i != self._layer_size - 2:
                activations[i + 1] = self._activate(activations[i + 1])
        activations[self._layer_size - 1] = self._activate_out_layer(activations[self._layer_size - 1])
        return activations

    def _mul_derivative(self, delta, activation):
        if self._activation == 'ReLU':
            delta[activation <= 0] = 0
        elif self._activation == 'sigmoid':
            # TODO
            pass

    def _get_grad(self, layer, activations, deltas, weight_grads, bias_grads):
        weight_grads[layer] = self._learning_rate * deltas[layer + 1].sum(axis=0)
        weight_grads[layer] = np.multiply(weight_grads[layer],
                                          activations[layer].sum(axis=0).reshape(self._layer_units[layer], 1))
        bias_grads[layer] = np.mean(deltas[layer + 1], axis=0)
        return weight_grads, bias_grads

    def _backprop(self, inp, oup, activations, deltas, weight_grads, bias_grads):
        activations[0] = inp
        activations = self._forward_pass(activations)
        if self._is_classification:
            deltas[-1] = np.multiply(oup - activations[-1],
                                     np.multiply(activations[-1], 1 - activations[-1]))
        else:
            deltas[-1] = oup - activations[-1]
        # TODO 分类需要改变
        for i in range(self._layer_size - 1, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self._weight[i - 1].T)
            self._mul_derivative(deltas[i - 1], activations[i - 1])
            weight_grads, bias_grads = self._get_grad(i - 1, activations, deltas, weight_grads, bias_grads)
        return weight_grads, bias_grads

    def _get_updates(self, weight_grads, bias_grads):
        coef = self._learning_rate * self._alpha
        weight_grads = [weight_grads[i] - self._weight[i] * coef for i in range(self._layer_size - 1)]
        bias_grads = [bias_grads[i] - self._bias_weight[i] * coef for i in range(self._layer_size - 1)]
        if self._momentum is not None:
            weight_grads = [x + self._momentum * y for x, y in zip(weight_grads, self._velocities_weight)]
            bias_grads = [x + self._momentum * y for x, y in zip(bias_grads, self._velocities_bias_weight)]
        weight_grads, bias_grads = self._get_clip_gradient(weight_grads, bias_grads)
        self._velocities_weight = copy.deepcopy(weight_grads)
        self._velocities_bias_weight = copy.deepcopy(bias_grads)
        return weight_grads, bias_grads

    def _update_weight(self, weight_grads, bias_grads):
        weight_updates, bias_updates = self._get_updates(weight_grads, bias_grads)
        for i in range(self._layer_size - 1):
            self._weight[i] += weight_updates[i]
            self._bias_weight[i] += bias_updates[i]

    def _train_mbgd(self, inp, oup, test_inp, test_oup):
        last_error, error = 110, 100
        self._velocities_weight = [np.zeros_like(weight) for weight in self._weight]
        self._velocities_bias_weight = [np.zeros_like(bias) for bias in self._bias_weight]
        activations = [np.zeros(shape=(size, 1)) for size in self._layer_units]
        deltas = [np.zeros(shape=(size, 1)) for size in self._layer_units]
        weight_grads = [np.zeros_like(weight) for weight in self._weight]
        bias_grads = [np.zeros_like(bias) for bias in self._bias_weight]
        for it in range(self._max_iter):
            for batch_slice in self._gen_batches(len(inp), self._batch_size):
                weight_grads, bias_grads = self._backprop(inp[batch_slice], oup[batch_slice], activations, deltas,
                                                          weight_grads,
                                                          bias_grads)
                self._update_weight(weight_grads, bias_grads)

            last_error = error
            error = self._get_error(test_inp, test_oup)
            if self._is_stop(last_error - error):
                break
            if self._is_print:
                print('iter: ', it)
                print('error: ', error)

    def _is_stop(self, difference):
        if difference > self._tol:
            return False
        if self._learning_rate > 1e-6:
            self._learning_rate /= 5
            return False
        else:
            return True

    def _get_error(self, test_inp, test_oup):
        predict = self._train_predict(test_inp)
        return ((predict - test_oup) ** 2).sum() / 2.

    def _train_predict(self, x):
        return np.array([self._get_ans(k) for k in x])

    @staticmethod
    def _get_sigmoid(net):
        """
        sigmoid激活函数
        """
        return 1. / (1. + np.exp(-net))

    @staticmethod
    def _get_ReLU(net):
        """
        ReLU激活函数
        """
        net[net < 0] = 0
        return net

    def _activate_out_layer(self, activations):
        if self._is_classification:
            return self._get_sigmoid(activations)
        else:
            return activations

    def _activate(self, x):
        if self._activation == 'ReLU':
            return self._get_ReLU(x)
        elif self._activation == 'sigmoid':
            return self._get_sigmoid(x)
        else:
            return x

    def _get_ans(self, inp):
        for weight, bias in zip(self._weight, self._bias_weight):
            net = np.add(np.dot(inp, weight), bias)
            inp = self._activate(copy.deepcopy(net))
        return net.T[0]

    def _normalize_x(self, x):
        """
        归一化数据
        :param x: input
        :return: Normalized x
        """
        self._x_dimension = x.shape[1]
        self._x_max = [x[:, i].max() for i in range(0, self._x_dimension)]
        self._x_min = [x[:, i].min() for i in range(0, self._x_dimension)]
        for i in range(0, self._x_dimension):
            x[:, i] = (x[:, i] - self._x_min[i]) / (self._x_max[i] - self._x_min[i])
        return x

    def _normalize_y(self, y):
        """
        归一化数据
        :param y: output
        :return: Normalized y
        """
        self._y_dimension = y.shape[1]
        self._y_max = [y[:, i].max() for i in range(0, self._y_dimension)]
        self._y_min = [y[:, i].min() for i in range(0, self._y_dimension)]
        for i in range(0, self._y_dimension):
            y[:, i] = (y[:, i] - self._y_min[i]) / (self._y_max[i] - self._y_min[i])
        return y

    def fit(self, x, y):
        """
        fit data
        :param x: input
        :param y: output
        :return: None
        """
        if len(x.shape) == 1:
            x = x.reshape(y.shape[0], 1)
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        if self._normalize:
            x, y = self._normalize_x(x), self._normalize_y(y)
        self._init_weight(x, y)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=1)
        self._batch_size = np.clip(self._batch_size, 1, len(X_train))
        if self._solver == 'mbgd':
            self._train_mbgd(X_train, y_train, X_test, y_test)
        else:
            print('no solver')
            return

    def predict(self, x):
        """
        预测数据
        :param x: input
        :return: result
        """
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if self._normalize:
            for i in range(0, self._x_dimension):
                x[:, i] = (x[:, i] - self._x_min[i]) / (self._x_max[i] - self._x_min[i])
        y = np.array([self._get_ans(k) for k in x])
        if self._normalize:
            for i in range(0, self._y_dimension):
                y[:, i] = y[:, i] * (self._y_max[i] - self._y_min[i]) + self._y_min[i]
        if y.shape[1] == 1:
            y = y.reshape(y.shape[0])
        return y

import copy
import random

import numpy as np
import math

from sklearn import model_selection


class BPNeuralNetwork:
    def __init__(self, solver='bgd', activation='ReLU',
                 init_method='normal', normalize=False,
                 learning_rate=1e-3, momentum=None,
                 clip_gradient=5, how_clip=None, is_print=False,
                 goal=1e-4, tol=1e-4, alpha=1e-4, max_iter=500,
                 batch_size=32,
                 hidden_layer_sizes=(100,)):
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
            self.__layer_size = 1
            self.__layer_number = (hidden_layer_sizes,)
        else:
            self.__layer_size = len(hidden_layer_sizes)
            self.__layer_number = hidden_layer_sizes
        self.__solver = solver
        self.__activation = activation
        self.__batch_size = batch_size
        self.__init_method = init_method
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__goal = goal
        self.__tol = tol
        self.__alpha = alpha
        self.____normalize = normalize
        self.__momentum = momentum
        self.__clip_gradient = clip_gradient
        self.__how_clip = how_clip
        self.__is_print = is_print

    def __init_layer_and_next_size(self):
        self.__layer_and_next_size = [(self.__layer_number[0], self.__input_size)]
        self.__layer_and_next_size.extend(
            [(self.__layer_number[i + 1], self.__layer_number[i]) for i in range(0, self.__layer_size - 1)])
        self.__layer_and_next_size.append((self.__output_size, self.__layer_number[self.__layer_size - 1]))

    def __init_weight_layer(self, layer_size):
        if self.__init_method == 'Xavier':
            scale = np.sqrt(6. / (layer_size[0] + layer_size[1]))
        elif self.__init_method == 'MSRA':
            scale = np.sqrt(2. / layer_size[1])
        elif self.__init_method == 'normal' or True:
            scale = 1
        if self.__init_method == 'Xavier':
            return np.random.uniform(-scale, scale, size=layer_size)
        elif self.__init_method == 'MSRA':
            return np.random.normal(0, scale, size=layer_size)
        elif self.__init_method == 'normal' or True:
            return np.random.normal(0, scale, size=layer_size)
            # return np.random.uniform(-scale, scale, size=layer_size)

    def __init_weight(self, inp, oup):
        """
        :param inp: 输入
        :param oup: 输出
        :return: None
        """
        self.__input_size = inp.shape[1]
        self.__output_size = oup.shape[1]
        self.__init_layer_and_next_size()
        self.__weight = [self.__init_weight_layer(size) for size in self.__layer_and_next_size]
        self.__bias_weight = [np.random.normal(0, 1, size=(size[0], 1)) for size in self.__layer_and_next_size]
        # self.__bias_weight = [self.__init_weight_layer(size=(size[0], 1)) for size in self.__layer_and_next_size]

    def __train_get_ans(self, x):
        """
        前向传播得到预测结果
        :param x: 输入
        :return: last_net 最后一层神经元的输入和
                o 神经元经过激活后的结果
                a 神经元之间的传递
        """
        x = np.array([x]).reshape(x.shape[0], 1)
        final_net, o, a = None, list(), list()
        for weight, bias in zip(self.__weight, self.__bias_weight):
            a.append(np.multiply(weight, x.T))
            x = np.add(np.dot(weight, x), bias)
            final_net = x
            x = self.__activate(copy.deepcopy(x))
            o.append(x)
        return final_net, o, a

    def __train_get_theta(self, final_net, o, t):
        """
         :param final_net: 最后一层神经元的输入和
         :param o: 每层神经元经过激活的结果
         :param t: 目标输出
         :return: 迭代中每层误差
         """
        t = np.array([t]).reshape(t.shape[0], 1)
        theta = [t - final_net]  # 因为最后一层是线性层
        for w, ou in zip(reversed(self.__weight[1:]), reversed(o[:-1])):
            if self.__activation == 'ReLU':
                ou[ou > 0] = 1
                ou[ou <= 0] = 0
                theta.append(np.multiply(np.dot(w.T, theta[-1]), ou))
            elif self.__activation == 'sigmoid':
                theta.append(np.multiply(np.dot(w.T, theta[-1]),
                                         np.multiply(ou, 1 - ou)))
        theta.reverse()
        return theta

    def __train_get_bias_delta(self, theta):
        """
        :param theta: 本次迭代中每层的误差
        :return: None
        """
        return [np.multiply(x, y) for x, y in zip(theta, self.__bias_weight)]

    @staticmethod
    def __train_get_delta(theta, a):
        return [np.multiply(x, y) for x, y in zip(theta, a)]

    def __clip_by_norm(self, delta):
        t = sum([x * x for x in delta[0]])
        if t < self.__clip_gradient:
            return delta
        else:
            return delta * (self.__clip_gradient / t)

    def __clip_by_global_norm(self, delta):
        t = 0
        for x in delta:
            t += sum(y * y for y in x[0])
        if t < self.__clip_gradient:
            return delta
        else:
            scale = self.__clip_gradient / t
            return [scale * x for x in delta]

    def __get_clip_gradient(self, delta, bias_delta):
        if self.__how_clip == 'normal':
            delta = [self.__clip_by_norm(k * self.__learning_rate / self.__batch_size) for k in delta]
            bias_delta = [self.__clip_by_norm(k * self.__learning_rate / self.__batch_size) for k in bias_delta]
        elif self.__how_clip == 'global':
            delta = self.__clip_by_global_norm([k * self.__learning_rate / self.__batch_size for k in delta])
            bias_delta = self.__clip_by_global_norm([k * self.__learning_rate / self.__batch_size for k in bias_delta])
        elif self.__how_clip is None or True:
            delta = [k * self.__learning_rate / self.__batch_size for k in delta]
            bias_delta = [k * self.__learning_rate / self.__batch_size for k in bias_delta]
        return delta, bias_delta

    def __train_bgd(self, inp, oup, test_inp, test_oup):
        """
        批梯度下降
        :param inp: 输入
        :param oup: 目标输出
        :param test_inp: 测试输入
        :param test_oup: 测试目标输出
        :return: None
        """
        error_difference, last_error, error, iter = 10, 110, 100, -1
        regularization_corr = self.__learning_rate * self.__alpha / self.__batch_size
        last_bias_delta = [np.zeros((size[0], 1)) for size in self.__layer_and_next_size]
        last_delta = [np.zeros(size) for size in self.__layer_and_next_size]
        self.__max_iter = len(inp) * self.__max_iter / self.__batch_size
        while ((error > self.__goal) or (error < self.__goal and error_difference > self.__tol)) \
                and iter < self.__max_iter:
            iter = iter + 1
            delta = [np.zeros(size) for size in self.__layer_and_next_size]
            bias_delta = [np.zeros((size[0], 1)) for size in self.__layer_and_next_size]
            for index in np.random.choice(range(len(inp)), size=self.__batch_size, replace=False):
                i = inp[index]
                j = oup[index]
                final_net, o, a = self.__train_get_ans(i.T)
                theta = self.__train_get_theta(final_net, o, j)
                delta = [x + y for x, y in zip(delta, self.__train_get_delta(theta, a))]
                bias_delta = [x + y for x, y in zip(bias_delta, self.__train_get_bias_delta(theta))]
            if self.__momentum is not None:
                delta = [x + self.__momentum * y for x, y in zip(delta, last_delta)]
                bias_delta = [x + self.__momentum * y for x, y in zip(bias_delta, last_bias_delta)]
            delta, bias_delta = self.__get_clip_gradient(delta, bias_delta)
            for delta_w, bias_delta_w, idx in zip(delta, bias_delta, range(0, self.__layer_size + 1)):
                last_delta[idx] = delta_w - regularization_corr * self.__weight[idx]
                last_bias_delta[idx] = bias_delta_w - regularization_corr * self.__bias_weight[idx]
                self.__weight[idx] = self.__weight[idx] + last_delta[idx]
                self.__bias_weight[idx] = self.__bias_weight[idx] + last_bias_delta[idx]

            last_error = error
            error = self.__get_error(test_inp, test_oup)
            error_difference = error - last_error
            if self.__is_print:
                print('iter: ', iter)
                print('error: ', error)

    def __get_error(self, test_inp, test_oup):
        predict = self.__train_predict(test_inp)
        return ((predict - test_oup) ** 2).sum() / 2.
        # return ((predict - test_oup) ** 2).sum() / len(test_oup)

    def __train_predict(self, x):
        """
        用于计算error的预测函数，不需要归一化回去
        :param x: 输入
        :return: 预测结果
        """
        return np.array([self.__get_ans(k.T) for k in x])

    @staticmethod
    def __get_sigmoid(net):
        """
        sigmoid激活函数
        """
        return 1. / (1. + np.exp(-net))
        # ret = list()
        # for k in net[0]:
        #     try:
        #         item = 1. / (1. + math.exp(-k))
        #     except OverflowError:
        #         item = 0
        #     ret.append(copy.deepcopy(item))
        # return ret

    @staticmethod
    def __get_ReLU(net):
        """
        ReLU激活函数
        """
        net[net < 0] = 0
        return net

    def __activate(self, x):
        if self.__activation == 'ReLU':
            x = self.__get_ReLU(x)
        elif self.__activation == 'sigmoid':
            x = self.__get_sigmoid(x)
        return x

    def __get_ans(self, inp):
        """
        预测结果
        :param inp: 输入
        :return: 输出结果
        """
        inp = np.array([inp]).reshape(inp.shape[0], 1)
        for weight, bias in zip(self.__weight, self.__bias_weight):
            net = np.add(np.dot(weight, inp), bias)
            inp = self.__activate(copy.deepcopy(net))
        return net.T[0]

    def __normalize_x(self, x):
        """
        归一化数据
        :param x: input
        :return: Normalized x
        """
        self.__x_dimension = x.shape[1]
        self.__x_max = [x[:, i].max() for i in range(0, self.__x_dimension)]
        self.__x_min = [x[:, i].min() for i in range(0, self.__x_dimension)]
        for i in range(0, self.__x_dimension):
            x[:, i] = (x[:, i] - self.__x_min[i]) / (self.__x_max[i] - self.__x_min[i])
        return x

    def __normalize_y(self, y):
        """
        归一化数据
        :param y: output
        :return: Normalized y
        """
        self.__y_dimension = y.shape[1]
        self.__y_max = [y[:, i].max() for i in range(0, self.__y_dimension)]
        self.__y_min = [y[:, i].min() for i in range(0, self.__y_dimension)]
        for i in range(0, self.__y_dimension):
            y[:, i] = (y[:, i] - self.__y_min[i]) / (self.__y_max[i] - self.__y_min[i])
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
        if self.____normalize:
            x, y = self.__normalize_x(x), self.__normalize_y(y)
        self.__init_weight(x, y)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=1)
        self.__batch_size = min(self.__batch_size, len(X_train))
        if self.__solver == 'bgd':
            self.__train_bgd(X_train, y_train, X_test, y_test)
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
        if self.____normalize:
            for i in range(0, self.__x_dimension):
                x[:, i] = (x[:, i] - self.__x_min[i]) / (self.__x_max[i] - self.__x_min[i])
        y = np.array([self.__get_ans(k.T) for k in x])
        if self.____normalize:
            for i in range(0, self.__y_dimension):
                y[:, i] = y[:, i] * (self.__y_max[i] - self.__y_min[i]) + self.__y_min[i]
        if y.shape[1] == 1:
            y = y.reshape(y.shape[0])
        return y

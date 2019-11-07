import numpy as np
import math

from sklearn import model_selection


class BPNeuralNetwork:
    def __init__(self, solver='sgd', activation='ReLU', learning_rate=1e-3,
                 clip_gradient=5, how_clip='none', is_print=False,
                 goal=1e-4, tol=1e-4, alpha=1e-4, max_iter=200,
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
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__goal = goal
        self.__tol = tol
        self.__alpha = alpha
        self.__clip_gradient = clip_gradient
        self.__how_clip = how_clip
        self.__is_print = is_print

    def __init_weight(self, inp, oup):
        """
        :param inp: 输入
        :param oup: 输出
        :return: None
        """
        # 初始化参数
        self.__input_size = inp.shape[1]
        self.__output_size = oup.shape[1]
        self.__weight = list()
        scale = np.sqrt(2 / self.__input_size)
        layer = np.random.normal(0, scale, size=(self.__input_size, self.__layer_number[0]))
        self.__weight.append(layer)
        for i in range(0, self.__layer_size - 1):
            scale = np.sqrt(2 / self.__layer_number[i])
            layer = np.random.normal(0, scale, size=(self.__layer_number[i], self.__layer_number[i + 1]))
            self.__weight.append(layer)
        scale = np.sqrt(2 / self.__layer_number[self.__layer_size - 1])
        layer = np.random.normal(0, scale, size=(self.__layer_number[self.__layer_size - 1], self.__output_size))
        self.__weight.append(layer)
        # 初始化偏置权值的参数
        self.__bias_weight = list()
        scale = np.sqrt(2 / 1)
        for i in range(0, self.__layer_size):
            layer = np.random.normal(0, scale, size=(1, self.__layer_number[i]))
            self.__bias_weight.append(layer)
        layer = np.random.normal(0, scale, size=(1, self.__output_size))
        self.__bias_weight.append(layer)

    def __clip_by_norm(self, delta):
        t = sum([x * x for x in delta.tolist()[0]])
        return delta * (self.__clip_gradient / max(t, self.__clip_gradient))

    def __clip_by_global_norm(self, delta):
        t = 0
        for x in delta:
            t += sum(y * y for y in x.tolist()[0])
        if self.__clip_gradient > t:
            return delta
        else:
            scale = self.__clip_gradient / t
            return [scale * x for x in delta]

    def __train_update_bias(self, theta):
        """
        :param theta: 本次迭代中每层的误差
        :return: None
        """
        bias_theta = list()
        if self.__how_clip == 'global':
            bias_theta = self.__clip_by_global_norm(
                [np.multiply(x, y) * self.__learning_rate for x, y in zip(theta, self.__bias_weight)])
        elif self.__how_clip == 'normal':
            bias_theta = [self.__clip_by_norm(np.multiply(x, y) * self.__learning_rate) for x, y in
                          zip(theta, self.__bias_weight)]
        elif self.__how_clip == 'none':
            bias_theta = [np.multiply(x, y) * self.__learning_rate for x, y in zip(theta, self.__bias_weight)]
        for delta, idx in zip(bias_theta, range(0, self.__layer_size + 1)):
            self.__bias_weight[idx] = self.__bias_weight[idx] + delta

    def __train_get_theta(self, net, o, t):
        """
        :param net: 每层神经元的输入和
        :param o: 每层神经元经过激活的结果
        :param t: 目标输出
        :return: 迭代中每层误差
        """
        theta = [t - net[-1]]  # 因为最后一层是线性层
        for w, ou in zip(reversed(self.__weight[1:]), reversed(o[:-1])):
            if self.__activation == 'sigmoid':
                theta.append(np.multiply(np.dot(theta[-1], w.T),
                                         np.multiply(ou, 1 - ou)))
            elif self.__activation == 'ReLU':
                theta.append(np.multiply(np.dot(theta[-1], w.T),
                                         np.array([1 if x > 0 else 0 for x in ou.tolist()[0]])))
        theta.reverse()
        return theta

    def __train_get_ans(self, x):
        """
        前向传播得到预测结果
        :param x: 输入
        :return: net 神经元的输入和
                o 神经元经过激活后的结果
                a 神经元之间的传递
        """
        net, o, a = list(), list(), list()
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        for weight, bias in zip(self.__weight, self.__bias_weight):
            a.append(np.multiply(x.T, weight))
            x = np.dot(x, weight)
            x = np.add(x, bias)  # 加上偏置
            net.append(x)
            if self.__activation == 'sigmoid':
                x = self.__get_sigmoid(x)
            elif self.__activation == 'ReLU':
                x = self.__get_ReLU(x)
            o.append(x)
        return net, o, a

    def __train_sgd(self, inp, oup, test_inp, test_oup):
        """
        随机梯度下降
        :param inp: 输入
        :param oup: 目标输出
        :param test_inp: 测试输入
        :param test_oup: 测试目标输出
        :return: None
        """
        last_error, error, iter = 110, 100, -1
        while (last_error - error > self.__tol or error > self.__goal) and iter < self.__max_iter:
            iter = iter + 1
            for i, j in zip(inp, oup):
                net, o, a = self.__train_get_ans(i)
                theta = self.__train_get_theta(net, o, j)
                self.__train_update_bias(theta)
                delta = list()
                if self.__how_clip is None:
                    delta = [np.multiply(x, y) * self.__learning_rate for x, y in zip(theta, a)]
                elif self.__how_clip == 'global':
                    delta = self.__clip_by_global_norm(
                        [np.multiply(x, y) * self.__learning_rate for x, y in zip(theta, a)])
                elif self.__how_clip == 'normal':
                    delta = [self.__clip_by_norm(np.multiply(x, y) * self.__learning_rate) for x, y in
                             zip(theta, a)]
                for delta_w, idx in zip(delta, range(0, self.__layer_size + 1)):
                    self.__weight[idx] = self.__weight[idx] \
                                         + delta_w \
                                         - self.__learning_rate * self.__alpha * self.__weight[idx]
            last_error = error
            error = self.__get_error(test_inp, test_oup)

            if self.__is_print:
                print('iter: ', iter)
                print('error: ', error)

    def __get_error(self, test_inp, test_oup):
        predict = self.__train_predict(test_inp)
        return ((predict - test_oup) ** 2).sum() / 2.

    @staticmethod
    def __get_sigmoid(net):
        """
        sigmoid激活函数
        """
        return 1. / (1. + np.exp(-net))
        # ret = list()
        # for k in net.tolist()[0]:
        #     try:
        #         item = 1. / (1. + math.exp(-k))
        #     except OverflowError:
        #         item = 0
        #     ret.append(item)
        # return ret

    @staticmethod
    def __get_ReLU(net):
        """
        ReLU激活函数
        """
        net[net < 0] = 0
        return net

    def __get_ans(self, inp):
        """
        预测结果
        :param inp: 输入
        :return: 输出结果
        """
        for weight, bias in zip(self.__weight, self.__bias_weight):
            net = np.dot(inp, weight)
            net = np.add(net, bias)
            if self.__activation == 'sigmoid':
                inp = self.__get_sigmoid(net)
            elif self.__activation == 'ReLU':
                inp = self.__get_ReLU(net)
        return net.tolist()[0]

    def __train_predict(self, x):
        """
        用于计算error的预测函数，不需要归一化回去
        :param x: 输入
        :return: 预测结果
        """
        return np.array([self.__get_ans(k) for k in x])

    def __normalize_x(self, x):
        """
        归一化数据
        :param x: input
        :param y: output
        :return: Normalized x
        """

        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        self.__x_dimension = x.shape[1]
        self.__x_max = [x[:, index].max() for index in range(0, self.__x_dimension)]
        self.__x_min = [x[:, index].min() for index in range(0, self.__x_dimension)]
        for index in range(0, self.__x_dimension):
            x[:, index] = (x[:, index] - self.__x_min[index]) / (self.__x_max[index] - self.__x_min[index])
        return x

    def __normalize_y(self, y):
        """
        归一化数据
        :param x: input
        :param y: output
        :return: Normalized y
        """

        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        self.__y_dimension = y.shape[1]
        self.__y_max = [y[:, index].max() for index in range(0, self.__y_dimension)]
        self.__y_min = [y[:, index].min() for index in range(0, self.__y_dimension)]
        for index in range(0, self.__y_dimension):
            y[:, index] = (y[:, index] - self.__y_min[index]) / (self.__y_max[index] - self.__y_min[index])
        return y

    def fit(self, x, y):
        """
        fit data
        :param x: input
        :param y: output
        :return: None
        """

        x, y = self.__normalize_x(x), self.__normalize_y(y)
        self.__init_weight(x, y)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=1)
        if self.__solver == 'sgd':
            self.__train_sgd(X_train, y_train, X_test, y_test)

    def predict(self, x):
        """
        预测数据
        :param x: input
        :return: result
        """

        x = self.__normalize_x(x)
        y = np.array([self.__get_ans(k) for k in x])
        for index in range(0, self.__y_dimension):
            y[:, index] = y[:, index] * (self.__y_max[index] - self.__y_min[index]) + self.__y_min[index]
        if self.__y_dimension == 1:
            y = y.reshape(y.shape[0])
        return y

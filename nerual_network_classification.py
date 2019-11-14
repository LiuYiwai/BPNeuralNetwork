from nerual_network import *


class BPNeuralNetworkClassification(BPNeuralNetwork):

    def __init__(self, solver='mbgd', activation='ReLU',
                 init_method='normal', normalize=False,
                 learning_rate=1e-3, momentum=None,
                 clip_gradient=5, how_clip=None, is_print=False,
                 goal=1e-4, tol=1e-4, alpha=1e-4, max_iter=500,
                 batch_size=32,
                 hidden_layer_sizes=(100,)):
        super().__init__(solver=solver, activation=activation,
                         init_method=init_method, normalize=normalize,
                         learning_rate=learning_rate, momentum=momentum,
                         clip_gradient=clip_gradient, how_clip=how_clip,
                         is_print=is_print, goal=goal, tol=tol, alpha=alpha,
                         max_iter=max_iter, batch_size=batch_size,
                         hidden_layer_sizes=hidden_layer_sizes, is_classification=True)

    def _get_ans(self, inp):
        """
        预测结果
        :param inp: 输入
        :return: 输出结果
        """
        ret = super(BPNeuralNetworkClassification, self)._get_ans(inp)
        return self._get_sigmoid(ret)

    def predict(self, x):
        """
        预测数据
        :param x: input
        :return: result
        """
        y = super(BPNeuralNetworkClassification, self).predict(x)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        return y

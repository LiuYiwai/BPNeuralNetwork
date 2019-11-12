from nerual_network import *


class BPNeuralNetworkRegression(BPNeuralNetwork):
    def _train_get_theta(self, final_net, o, t):
        """
         :param final_net: 最后一层神经元的输入和
         :param o: 每层神经元经过激活的结果
         :param t: 目标输出
         :return: 迭代中每层误差
         """
        t = np.array([t]).reshape(t.shape[0], 1)
        theta = [t - final_net]  # 因为最后一层是线性层
        for w, ou in zip(reversed(self._weight[1:]), reversed(o[:-1])):
            if self._activation == 'ReLU':
                ou[ou > 0] = 1
                ou[ou <= 0] = 0
                theta.append(np.multiply(np.dot(w.T, theta[-1]), ou))
            elif self._activation == 'sigmoid':
                theta.append(np.multiply(np.dot(w.T, theta[-1]),
                                         np.multiply(ou, 1 - ou)))
        theta.reverse()
        return theta

    def _get_ans(self, inp):
        """
        预测结果
        :param inp: 输入
        :return: 输出结果
        """
        ret = super(BPNeuralNetworkRegression, self)._get_ans(inp)
        return ret

    def predict(self, x):
        """
        预测数据
        :param x: input
        :return: result
        """
        y = super(BPNeuralNetworkRegression, self).predict(x)
        return y
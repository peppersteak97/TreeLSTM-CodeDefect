import tensorflow as tf
import numpy as np

class TreeLSTMCell:

    # 定义TreeLSTM的Cell类

    def _init_(self):
        # 输入的节点特征向量，由ast2vec得到，维度不确定
        self.inputs_wt = tf.placeholder(tf.float32, [None, None],"inputs")

    #  实例化之后调用一次
    def weight_variable(self, dim, init=tf.random_normal_initializer(mean=0, stddev=1)):
        with tf.variable_scope("lstm_cell"):
            # Forget Gate 可训练参量
            w_for = tf.get_variable("forget_w", shape=dim, initializer=init)
            u_for = tf.get_variable("forget_u", shape=dim, initializer=init)
            b_for = tf.get_variable("forget_b", shape=dim, initializer=init)
            # Input Gate 可训练参量
            w_in = tf.get_variable("input_w", shape=dim, initializer=init)
            u_in = tf.get_variable("input_u", shape=dim, initializer=init)
            b_in = tf.get_variable("input_b", shape=dim, initializer=init)
            w_ce = tf.get_variable("input_ce_w", shape=dim, initializer=init)
            u_ce = tf.get_variable("input_ce_u", shape=dim, initializer=init)
            b_ce = tf.get_variable("input_ce_b", shape=dim, initializer=init)
            # Output Gate 可训练参量
            w_out = tf.get_variable("output_w", shape=dim, initializer=init)
            u_out = tf.get_variable("output_u", shape=dim, initializer=init)
            b_out = tf.get_variable("output_b", shape=dim, initializer=init)

    def init_inputs(self, np_hc):
        # np_hc是tensor数组
        with tf.variable_scope("lstm_cell", reuse=True):
            # Forget Gate 可训练参量
            w_for = tf.get_variable("forget_w")
            u_for = tf.get_variable("forget_u")
            b_for = tf.get_variable("forget_b")
        # 获取h的维度
        h_dim = np_hc[0][0].shape[0].value
        self.f = tf.Variable(tf.zeros(h_dim, dtype=tf.float32))
        self.h = tf.Variable(tf.zeros(h_dim, dtype=tf.float32))
        # 这里的np_hc为numpy数组
        for hc_pair in np_hc.flat:
            tmp_f = tf.nn.sigmoid(tf.multiply(w_for, self.inputs_wt) + tf.multiply(u_for, hc_pair[0]) + b_for)
            self.f = tf.add(self.f, tf.multiply(tmp_f, hc_pair[1]))
            self.h = tf.add(self.h, hc_pair[0])

    def lstm_cell(self):
        # 定义cell内部的运算
        with tf.variable_scope("lstm_cell", reuse=True):
            # Input Gate 可训练参量
            w_in = tf.get_variable("input_w")
            u_in = tf.get_variable("input_u")
            b_in = tf.get_variable("input_b")
            w_ce = tf.get_variable("input_ce_w")
            u_ce = tf.get_variable("input_ce_u")
            b_ce = tf.get_variable("input_ce_b")
            # Output Gate 可训练参量
            w_out = tf.get_variable("output_w")
            u_out = tf.get_variable("output_u")
            b_out = tf.get_variable("output_b")
        it = tf.nn.sigmoid(tf.multiply(w_in, self.inputs_wt) + tf.multiply(u_in, self.h) + b_in)
        ct = tf.nn.tanh(tf.multiply(w_ce, self.inputs_wt) + tf.multiply(u_ce, self.h) + b_ce)
        ct = tf.add(tf.multiply(it, ct), self.f)
        ot = tf.nn.sigmoid(tf.multiply(w_out, self.inputs_wt) + tf.multiply(u_out, self.h) + b_out)
        ht = tf.multiply(ot, tf.nn.tanh(ct))
        return np.array(ht, ct)


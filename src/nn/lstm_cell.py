import tensorflow as tf
import numpy as np
#from config import *

#h_dim = []

#c_dim = []

#word_vocab_size = 100

#word_embd_dim = 10


# Forget Gate 可训练参量
w_for = tf.Variable(tf.random_normal(h_dim))
u_for = tf.Variable(tf.random_normal(h_dim))
b_for = tf.Variable(tf.random_normal(h_dim))

# Input Gate 可训练参量
w_in = tf.Variable(tf.random_normal(h_dim))
u_in = tf.Variable(tf.random_normal(h_dim))
b_in = tf.Variable(tf.random_normal(h_dim))

w_ce = tf.Variable(tf.random_normal(h_dim))
u_ce = tf.Variable(tf.random_normal(h_dim))
b_ce = tf.Variable(tf.random_normal(h_dim))

# Output Gate 可训练参量
w_out = tf.Variable(tf.random_normal(h_dim))
u_out = tf.Variable(tf.random_normal(h_dim))
b_out = tf.Variable(tf.random_normal(h_dim))

# 定义embedding layer
#with tf.name_scope('word_embedding'):
#    W = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embd_dim]), name="W")
#    embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size, word_embd_dim])
#    embedding_init = W.assign(embedding_placeholder)
#    embd_fp_word = tf.nn.embedding_lookup(W, )


# 定义 LSTMCell
def TreeLSTMCell(inputs, wt):
    # 这里的inputs是子节点(h, c)的list，wt是抽象语法树对应的节点的向量
    # 返回该cell的h,c

    #h_dim = inputs[0][0].shape[0].value()

    h = tf.Variable(tf.zeros(h_dim))
    forget_out = tf.Variable(tf.zeros(h_dim))

    for hc_pair in inputs:
        # 对所有的子节点的输出h求和
        h = tf.add(h, hc_pair[0])
        # 对forget gate的输入求和
        f = tf.nn.sigmoid(tf.multiply(w_for, wt)+tf.multiply(u_for, hc_pair[0])+b_for)
        forget_out = tf.add(forget_out, tf.multiply(f, hc_pair[1]))

    it = tf.nn.sigmoid(tf.multiply(w_in, wt) + tf.multiply(u_in, h) + b_in)

    ct = tf.nn.tanh(tf.multiply(w_ce, wt) + tf.multiply(u_ce, h) + b_ce)

    ct= tf.add(tf.multiply(it, ct), forget_out)

    ot = tf.nn.sigmoid(tf.multiply(w_out, wt) + tf.multiply(u_out, h) + b_out)

    ht = tf.multiply(ot, tf.nn.tanh(ct))

    return [ht, ct]









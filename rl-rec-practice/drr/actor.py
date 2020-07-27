#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午5:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 连续动作的action,如果是推荐1个item的话,由于action的维度为19,则输出一个19维的向量,

import tensorflow as tf

class Actor(object):
    """policy function approximator"""

    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate, scope="actor"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.output_size = output_size#即item的维度
        self.weights_len = weights_len #1,currently only generate 1 item，动作中item的个数
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator actor network
            self.state, self.action_weights, self.len_seq = self._build_net("estimator_actor")
            self.network_params = tf.trainable_variables()
            # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_actor')

            # target actor network
            self.target_state, self.target_action_weights, self.target_len_seq = self._build_net("target_actor")
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]
            # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assign(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
            self.params_gradients = list(
                map(
                    lambda x: tf.div(x, self.batch_size * self.a_dim),
                    tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.a_dim]),
                                 self.network_params, -self.a_gradient)
                )
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.params_gradients, self.network_params)
            )
            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    @staticmethod
    def cli_value(x, v):
        x = tf.cast(x, tf.int64)
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            len_seq = tf.placeholder(tf.int32, [None])
            cell = tf.nn.rnn_cell.GRUCell(self.output_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer())
            outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            print('Actor outputs shape',outputs.shape)
            # outputs(None,1,15*19) len_seq(None)
            outputs1 = self._gather_last_output(outputs, len_seq)
            print('outputs1.shape',outputs1.shape)

            layer1 = tf.layers.Dense(64, activation=tf.nn.relu)(outputs1)
            layer2 = tf.layers.Dense(32, activation=tf.nn.relu)(layer1)
            outputs = tf.layers.Dense(self.output_size, activation=tf.nn.tanh)(layer2)
            print('outputs.shape',outputs.shape)

        return state, outputs, len_seq

    def train(self, state, a_gradient, len_seq):
        self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient, self.len_seq: len_seq})

    def predict(self, state, len_seq):
        return self.sess.run(self.action_weights, feed_dict={self.state: state, self.len_seq: len_seq})

    def predict_target(self, state, len_seq):
        return self.sess.run(self.target_action_weights, feed_dict={self.target_state: state,
                                                                    self.target_len_seq: len_seq})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
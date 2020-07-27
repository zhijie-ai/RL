#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/26 21:07                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

# Critic
class Critic():
    ''' Value function approximator. '''

    def __init__(self, sess, state_space_size, action_space_size, history_length, embedding_size, tau, learning_rate,
                 scope='critic'):
        self.sess = sess
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.history_length = history_length
        self.embedding_size = embedding_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Build Critic network
            self.critic_Q_value, self.state, self.action, self.sequence_length = self._build_net('estimator_critic')
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_critic')

            # Build target Critic network
            self.target_Q_value, self.target_state, self.target_action, self.target_sequence_length = self._build_net(
                'target_critic')
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

            # Initialize target network weights with network weights (θ^µ′ ← θ^µ)
            self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
                                               for i in range(len(self.target_network_params))]

            # Update target network weights (θ^µ′ ← τθ^µ + (1 − τ)θ^µ′)
            self.update_target_network_params = [self.target_network_params[i].assign(
                tf.multiply(self.tau, self.network_params[i]) +
                tf.multiply(1 - self.tau, self.target_network_params[i]))
                for i in range(len(self.target_network_params))]

            # Minimize MSE between Critic's and target Critic's outputed Q-values
            self.expected_reward = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.expected_reward, self.critic_Q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # Compute ∇_a.Q(s, a|θ^µ)
            self.action_gradients = tf.gradients(self.critic_Q_value, self.action)

    def _build_net(self, scope):
        ''' Build the (target) Critic network. '''

        def gather_last_output(data, seq_lens):
            def cli_value(x, v):
                y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
                return tf.where(tf.greater(x, y), x, y)

            this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
            tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
            indices = tf.stack([this_range, tmp_end], axis=1)
            return tf.gather_nd(data, indices)

        with tf.variable_scope(scope):
            # Inputs: current state, current action
            # Outputs: predicted Q-value
            state = tf.placeholder(tf.float32, [None, self.state_space_size], 'state')
            state_ = tf.reshape(state, [-1, self.history_length, self.embedding_size])
            action = tf.placeholder(tf.float32, [None, self.action_space_size], 'action')
            sequence_length = tf.placeholder(tf.int64, [None], name='critic_sequence_length')
            cell = tf.nn.rnn_cell.GRUCell(self.history_length,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer())
            predicted_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=sequence_length)
            predicted_state = gather_last_output(predicted_state, sequence_length)

            inputs = tf.concat([predicted_state, action], axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            critic_Q_value = tf.layers.Dense(1)(layer2)
            return critic_Q_value, state, action, sequence_length

    def train(self, state, action, sequence_length, expected_reward):
        ''' Minimize MSE between expected reward and target Critic's Q-value. '''
        return self.sess.run([self.critic_Q_value, self.loss, self.optimizer],
                             feed_dict={
                                 self.state: state,
                                 self.action: action,
                                 self.sequence_length: sequence_length,
                                 self.expected_reward: expected_reward})

    def predict(self, state, action, sequence_length):
        ''' Returns Critic's predicted Q-value. '''
        return self.sess.run(self.critic_Q_value,
                             feed_dict={
                                 self.state: state,
                                 self.action: action,
                                 self.sequence_length: sequence_length})

    def predict_target(self, state, action, sequence_length):
        ''' Returns target Critic's predicted Q-value. '''
        return self.sess.run(self.target_Q_value,
                             feed_dict={
                                 self.target_state: state,
                                 self.target_action: action,
                                 self.target_sequence_length: sequence_length})

    def get_action_gradients(self, state, action, sequence_length):
        ''' Returns ∇_a.Q(s, a|θ^µ). '''
        return np.array(self.sess.run(self.action_gradients,
                                      feed_dict={
                                          self.state: state,
                                          self.action: action,
                                          self.sequence_length: sequence_length})[0])

    def init_target_network(self):
        self.sess.run(self.init_target_network_params)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
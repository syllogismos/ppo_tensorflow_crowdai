"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""

import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, logger, snapshot=None):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._init_layers_sizes()
        print(snapshot, '$$$$$')
        if snapshot == None:
            print('@@@@ new value object')
            self._build_graph()
            self.sess = tf.Session(graph=self.g)
            self.sess.run(self.init)
        else:
            self._build_graph_from_snapshot(snapshot)
        self.value_checkpoint = logger.get_file_name('value-model')
        self.saver.save(self.sess, self.value_checkpoint, latest_filename='value_checkpoint', global_step=0)

    def _init_layers_sizes(self):
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        self.hid1_size = self.obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
        self.hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 1e-2 / np.sqrt(self.hid2_size)  # 1e-3 empirically determined
        print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(self.hid1_size, self.hid2_size, self.hid3_size, self.lr))

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, self.hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, self.hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.hid1_size)), name="h2")
            out = tf.layers.dense(out, self.hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

            tf.add_to_collection('obs_ph_chk', self.obs_ph)
            tf.add_to_collection('val_ph_chk', self.val_ph)
            tf.add_to_collection('out_chk', self.out)
            tf.add_to_collection('loss_chk', self.loss)
            tf.add_to_collection('train_op_chk', self.train_op)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=100)
        # self.sess = tf.Session(graph=self.g)
        # self.sess.run(self.init)

    def _build_graph_from_snapshot(self, snapshot):
        """ Build graph from snapshot provided """
        print('@@@@@@@@@@@@@@loading from value snapshot')
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        with self.g.as_default():
            latest_chkp_file = tf.train.latest_checkpoint(snapshot, latest_filename='value_checkpoint')
            meta_file = latest_chkp_file + '.meta'
            if not os.path.exists(meta_file):
                meta_file = snapshot + '/value-model-0.meta'
            meta_graph = tf.train.import_meta_graph(meta_file)
            self.saver = tf.train.Saver(max_to_keep=100)
            meta_graph.restore(self.sess, latest_chkp_file)
            self.obs_ph = tf.get_collection('obs_ph_chk')[0]
            self.val_ph = tf.get_collection('val_ph_chk')[0]
            self.out = tf.get_collection('out_chk')[0]
            self.loss = tf.get_collection('loss_chk')[0]
            self.train_op = tf.get_collection('train_op_chk')[0]


    def fit(self, x, y, logger, episode):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        self.saver.save(self.sess, self.value_checkpoint, global_step=episode, latest_filename='value_checkpoint')
        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

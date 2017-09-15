"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf
import os


class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, logger, snapshot=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
        """
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._init_layers_sizes()
        if snapshot == None:
            print('#### new policy object')
            self._build_graph()
            self._init_session()
        else:
            print('##################building from snapshot')
            self._build_graph_from_snapshot(snapshot)
        self.policy_checkpoint = logger.get_file_name('policy-model')
        self.saver.save(self.sess, self.policy_checkpoint, latest_filename='policy_checkpoint', global_step=0)

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=100)

    def _build_graph_from_snapshot(self, snapshot):
        """ Build graph from snapshot provided """
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        with self.g.as_default():
            latest_chkp_file = tf.train.latest_checkpoint(snapshot, latest_filename='policy_checkpoint')
            meta_file = latest_chkp_file + '.meta'
            if not os.path.exists(meta_file):
                meta_file = snapshot + '/policy-model-0.meta'
            meta_graph = tf.train.import_meta_graph(meta_file)
            self.saver = tf.train.Saver(max_to_keep=100)
            meta_graph.restore(self.sess, latest_chkp_file)

            self.obs_ph = tf.get_collection('obs_ph_chk')[0]
            self.act_ph = tf.get_collection('act_ph_chk')[0]
            self.advantages_ph = tf.get_collection('advantages_ph_chk')[0]
            self.beta_ph = tf.get_collection('beta_ph_chk')[0]
            self.eta_ph = tf.get_collection('eta_ph_chk')[0]
            self.lr_ph = tf.get_collection('lr_ph_chk')[0]
            self.old_log_vars_ph = tf.get_collection('old_log_vars_ph_chk')[0]
            self.old_means_ph = tf.get_collection('old_means_ph_chk')[0]

            self.means = tf.get_collection('means_chk')[0]
            self.log_vars = tf.get_collection('log_vars_chk')[0]

            self.logp = tf.get_collection('logp_chk')[0]
            self.logp_old = tf.get_collection('logp_old_chk')[0]

            self.kl = tf.get_collection('kl_chk')[0]
            self.entropy = tf.get_collection('entropy_chk')[0]

            self.sampled_act = tf.get_collection('sampled_act_chk')[0]

            self.loss = tf.get_collection('loss_chk')[0]
            self.train_op = tf.get_collection('train_op_chk')[0]

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

        tf.add_to_collection('obs_ph_chk', self.obs_ph)
        tf.add_to_collection('act_ph_chk', self.act_ph)
        tf.add_to_collection('advantages_ph_chk', self.advantages_ph)
        tf.add_to_collection('beta_ph_chk', self.beta_ph)
        tf.add_to_collection('eta_ph_chk', self.eta_ph)
        tf.add_to_collection('lr_ph_chk', self.lr_ph)
        tf.add_to_collection('old_log_vars_ph_chk', self.old_log_vars_ph)
        tf.add_to_collection('old_means_ph_chk', self.old_means_ph)

    def _init_layers_sizes(self):
        """ Hidden layers are sized based on obs dim """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        self.hid1_size = self.obs_dim * 10  # 10 empirically determined
        self.hid3_size = self.act_dim * 10  # 10 empirically determined
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(self.hid2_size)  # 9e-4 empirically determined

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
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
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / self.hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * self.hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0

# name tensors/operations that we need to restore from checkpoint
        tf.add_to_collection('means_chk', self.means)
        tf.add_to_collection('log_vars_chk', self.log_vars)

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(self.hid1_size, self.hid2_size, self.hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

        tf.add_to_collection('logp_chk', self.logp)
        tf.add_to_collection('logp_old_chk', self.logp_old)

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

        tf.add_to_collection('kl_chk', self.kl)
        tf.add_to_collection('entropy_chk', self.entropy)

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

        tf.add_to_collection('sampled_act_chk', self.sampled_act)
        # sample_act_2 = tf.multiply(tf.exp(self.log_vars / 2.0), 
        #         tf.random_normal(shape=(self.act_dim,)))
        # self.sampled_act = tf.add(self.means, sample_act_2, name='sampled_act')

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        loss1 = -tf.reduce_mean(self.advantages_ph *
                                tf.exp(self.logp - self.logp_old))
        loss2 = tf.reduce_mean(self.beta_ph * self.kl)
        loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

        tf.add_to_collection('loss_chk', self.loss)
        tf.add_to_collection('train_op_chk', self.train_op)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger, episode):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.saver.save(self.sess, self.policy_checkpoint, global_step=episode, latest_filename='policy_checkpoint')

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

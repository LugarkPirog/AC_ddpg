import tensorflow as tf
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    # TODO: Make prioritized
    def __init__(self, max_len=256):
        self.buffer = deque(maxlen=max_len)
        self.__maxlen = max_len

    @property
    def count(self):
        return len(self.buffer)

    def add(self, state, action, reward):
        self.buffer.extend((triplet for triplet in zip(state, action, reward)))

    def sample(self, num):
        return random.sample(self.buffer, num)

    def clear(self):
        self.buffer = deque(maxlen=self.__maxlen)


class OUNoise:
    def __init__(self, action_dim, mu=0., theta=.05, sigma=.05):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.act_dim = action_dim
        self.state = np.ones(self.act_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.act_dim) * self.mu

    def poshumim(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.act_dim)
        self.state = x + dx
        return self.state


class NormalNoise:
    def __init__(self, action_dim, mu, *args, decay=.99):
        self.mu = mu
        self.act_dim = action_dim
        self.step = 0
        self.decay = decay

    def poshumim(self):
        self.step += 1
        return np.random.randn(self.act_dim) * (self.mu*(self.decay**(self.step-1)))


class Network:
    def __init__(self, sess, name, state_dim, act_dim):
        self.sess = sess
        self.name = name
        self.state_dim = state_dim
        self.act_dim = act_dim


class Actor(Network):
    def __init__(self, sess, name, state_dim, act_dim, l1_s, l2_s, learning_rate, ema_decay=.99):
        super().__init__(sess, name, state_dim, act_dim)
        self.ema_decay = ema_decay
        self.input, \
        self.actions, \
        self.vars = self.create_network(l1_s, l2_s)

        self.q_grad_input, \
        self.update_step = self.create_updater(learning_rate)

        self.t_state_in, \
        self.t_actions, \
        self.t_upd, \
        self.t_vars = self.create_target_net()

        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def create_network(self, l1_size, l2_size):
        with tf.variable_scope(self.name):
            inp = tf.placeholder(tf.float32, (None, self.state_dim), name='state_input')
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w', (self.state_dim, l1_size))
                b1 = tf.Variable(tf.zeros((l1_size,)), name='b')
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w', (l1_size, l2_size))
                b2 = tf.Variable(tf.zeros((l2_size,)), name='b')
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                w3 = tf.get_variable('w', (l2_size, self.act_dim))
                b3 = tf.Variable(tf.zeros((self.act_dim,)), name='b')

            l1 = tf.nn.relu(tf.matmul(inp, w1) + b1)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            out_actions = tf.nn.sigmoid(tf.matmul(l2, w3) + b3, name='out_actions')

        net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return inp, out_actions, net

    def create_target_net(self):
        with tf.name_scope(self.name + '_shadow'):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_input')
            ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
            target_upd = ema.apply(self.vars)
            target_vars = [ema.average(x) for x in self.vars]

            l1 = tf.nn.relu(tf.matmul(state_in, target_vars[0]) + target_vars[1])
            l2 = tf.nn.relu(tf.matmul(l1, target_vars[2]) + target_vars[3])
            out = tf.nn.sigmoid(tf.matmul(l2, target_vars[4]) + target_vars[5])

        return state_in, out, target_upd, target_vars

    def update_target(self):
        self.sess.run(self.t_upd)

    def create_updater(self, lr):
        q_grad_inp = tf.placeholder(tf.float32, [None, self.act_dim], name='q_grad_in')
        net_grads = tf.gradients(self.actions, self.vars, q_grad_inp, name='q_grads')
        opt_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(net_grads, self.vars))
        return q_grad_inp, opt_step

    def get_actions(self, input_states):
        return self.sess.run(self.actions, feed_dict={self.input: input_states})

    def get_target_actions(self, input_states):
        return self.sess.run(self.t_actions, feed_dict={self.t_state_in: input_states})

    def apply_grads(self, grads, states, rewards):
        g = grads[0] * np.sign(rewards)
        self.sess.run(self.update_step, feed_dict={self.q_grad_input: g, self.input: states})


class Critic(Network):
    def __init__(self, sess, name, state_dim, act_dim, l1_size, l2_size, learning_rate, ema_decay=.99):
        super().__init__(sess, name, state_dim, act_dim)

        self.ema_decay = ema_decay
        self.state_input, \
        self.action_input, \
        self.q_value, \
        self.vars = self.create_network(l1_size, l2_size)

        self.q_target, \
        self.loss, \
        self.action_grads, \
        self.update_step = self.create_updater(learning_rate)

        self.t_state_in, \
        self.t_act_in, \
        self.t_q_value, \
        self.t_upd, \
        self.t_vars = self.create_target_net()

        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def create_network(self, l1_size, l2_size):
        with tf.variable_scope(self.name):
            state_inp = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
            action_input = tf.placeholder(tf.float32, [None, self.act_dim], name='action_input')
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', (self.state_dim, l1_size))
                b1 = tf.Variable(tf.zeros([l1_size, ]), name='b')
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', (l1_size, l2_size))
                w2_act = tf.get_variable('w2_act', (self.act_dim, l2_size))
                b2 = tf.Variable(tf.zeros([l2_size, ]), name='b')
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', (l2_size, 1))
                b3 = tf.Variable(tf.zeros([1, ]), name='b')

            l1 = tf.nn.relu(tf.matmul(state_inp, w1) + b1)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + tf.matmul(action_input, w2_act) + b2)
            q_value = tf.identity(tf.matmul(l2, w3) + b3, name='q_value')

        net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return state_inp, action_input, q_value, net

    def create_target_net(self):
        with tf.name_scope(self.name + '_shadow'):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_input')
            act_in = tf.placeholder(tf.float32, (None, self.act_dim), name='action_input')
            ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
            target_upd = ema.apply(self.vars)
            target_vars = [ema.average(x) for x in self.vars]

            l1 = tf.nn.relu(tf.matmul(state_in, target_vars[0]) + target_vars[1])
            l2 = tf.nn.relu(tf.matmul(l1, target_vars[2]) + tf.matmul(act_in, target_vars[3]) + target_vars[4])
            out = tf.identity(tf.matmul(l2, target_vars[5]) + target_vars[6])

        return state_in, act_in, out, target_upd, target_vars

    def update_target(self):
        self.sess.run(self.t_upd)

    def create_updater(self, lr):
        q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')
        loss = tf.reduce_mean(tf.square(q_target - self.q_value), name='critic_loss')
        action_gradients = tf.gradients(self.q_value, self.action_input, name='action_grads')
        tf.summary.histogram('action_grads', action_gradients)
        update_step = tf.train.AdamOptimizer(lr).minimize(loss)
        return q_target, loss, action_gradients, update_step

    def update(self, reward, states, actions):
        self.sess.run(self.update_step, feed_dict={self.q_target: reward,
                                                   self.state_input: states,
                                                   self.action_input: actions})

    def get_q(self, states, actions):
        return self.sess.run(self.q_value, feed_dict={self.state_input: states,
                                                      self.action_input: actions})

    def get_target_q(self, states, actions):
        return self.sess.run(self.t_q_value, feed_dict={self.t_state_in: states,
                                                        self.t_act_in: actions})

    def get_act_grads(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state_input: states,
                                                           self.action_input: actions}) * np.clip(np.mean(actions, axis=0),1e-6, 1.)


class DDPG:
    # TODO: Add gradient masking
    def __init__(self, max_buffer_len, batch_size, action_dim=60, state_dim=32,
                 noise_theta=.17, noise_sigma=.3, noise_mu=.3, noise='normal', actor_lr=1e-4, critic_lr=1e-3,
                 start_train_at=100, test_verbose=10, save_model_every=25,
                 savedir='./DDPG/model/', logdir='./DDPG/logs', name='ddpg'):
        self._start = start_train_at
        self.test_verbose = test_verbose
        self.sess = tf.Session()
        self.bs = batch_size
        self.gs = 0  # global step
        self.name = name
        self.savedir = savedir + self.name
        self.save_model_every = save_model_every
        # self.logdir = logdir

        self.actor = Actor(self.sess, 'actor', state_dim=state_dim,
                           act_dim=action_dim, l1_s=64, l2_s=128, learning_rate=actor_lr)
        self.critic = Critic(self.sess, 'critic', state_dim=state_dim,
                             act_dim=action_dim, l1_size=64, l2_size=64, learning_rate=critic_lr)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name[:-2], var)

        self.__summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(logdir)
        self.summary_writer.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.buffer = ReplayBuffer(max_buffer_len)
        if noise == 'ou':
            self.noize = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)
        elif noise == 'normal':
            self.noize = NormalNoise(action_dim, noise_mu)

    def train(self):
        train_batch = np.array(self.buffer.sample(self.bs))
        states = np.array(train_batch[:, 0].tolist())
        actions = np.array(train_batch[:, 1].tolist())
        rewards = train_batch[:, 2].reshape((-1, 1))

        self.critic.update(rewards, states, actions)

        #actions_new = self.actor.get_actions(states)
        q_grads = self.critic.get_act_grads(states, actions)
        self.actor.apply_grads(q_grads, states, rewards)

        self.actor.update_target()
        self.critic.update_target()

    def noised_actions(self, states):
        return self.actions(states) + self.noize.poshumim()

    def actions(self, states):
        return self.actor.get_actions(states)

    def step(self, states, actions, rewards):
        self.gs += 1
        self.buffer.add(states, actions, rewards)
        if self.buffer.count < self._start:
            return
        self.train()

        if self.gs % self.test_verbose == self.test_verbose - 1:
            self.write_summary(states, actions)
        if self.gs % self.save_model_every == self.save_model_every - 1:
            self.save_model()

    def save_model(self):
        self.saver.save(self.sess, self.savedir)

    def load_model(self):
        self.saver.restore(self.sess, self.savedir)

    def write_summary(self, states, acts):
        # TODO: add grad split to visualize individual grad behaviour
        s = self.sess.run(self.__summary, feed_dict={self.critic.action_input:acts, self.critic.state_input:states})
        self.summary_writer.add_summary(s, self.gs)


if __name__ == '__main__':
    ou = OUNoise(2, theta=.05, sigma=.05)
    states = []

    print(ou.poshumim())
    print(ou.poshumim())
    print(ou.poshumim())
    print(ou.poshumim())
    print(ou.poshumim())

    for i in range(1000):
        states.append(ou.poshumim())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
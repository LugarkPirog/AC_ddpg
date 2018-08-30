import tensorflow as tf
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_len=256):
        self.buffer = deque(maxlen=max_len)
        self.__maxlen = max_len

    @property
    def count(self):
        return len(self.buffer)

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, num):
        return np.random.choice(self.buffer, num, replace=False)

    def clear(self):
        self.buffer = deque(maxlen=self.__maxlen)

class OUNoise:
    def __init__(self, action_dim, mu=0., theta=.2, sigma=.2, seed=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.act_dim = action_dim
        self.state = np.ones(self.act_dim) * self.mu
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.act_dim)*self.mu

    def poshumim(self):
        dx = self.theta*(self.mu - self.state) + self.sigma*np.random.randn(len(self.state))
        self.state += dx
        return self.state



class Network:
    def __init__(self, sess, name, state_dim, act_dim):

        self.sess = sess
        self.name = name
        self.state_dim = state_dim
        self.act_dim = act_dim


class Actor(Network):
    def __init__(self, sess, name, state_dim, act_dim, l1_s, l2_s):
        super().__init__(sess, name, state_dim, act_dim)

        self.input,\
        self.actions,\
        self.vars = self.create_network(l1_s, l2_s)

        self.q_grad_input,\
        self.update_step = self.create_updater()

        # self.sess.run(tf.global_variables_initializer())

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

    def create_updater(self):
        q_grad_inp = tf.placeholder(tf.float32, [None, self.act_dim], name='q_grad_in')
        net_grads = tf.gradients(self.actions, self.vars, q_grad_inp, name='q_grads')
        opt_step = tf.train.AdamOptimizer(1e-3).apply_gradients(zip(net_grads, self.vars))
        return q_grad_inp, opt_step

    def get_actions(self, input_states):
        return self.sess.run(self.actions, feed_dict={self.input:input_states})

    def apply_grads(self, grads, states):
        self.sess.run(self.update_step, feed_dict={self.q_grad_input:grads, self.input:states})


class Critic(Network):
    def __init__(self, sess, name, state_dim, act_dim, l1_size, l2_size):
        super().__init__(sess, name, state_dim, act_dim)

        self.state_input,\
        self.action_input,\
        self.q_value,\
        self.action_grads,\
        self.vars = self.create_network(l1_size, l2_size)

        self.q_target,\
        self.loss,\
        self.update_step = self.create_updater()

        # self.sess.run(tf.global_variables_initializer())

    def create_network(self, l1_size, l2_size):
        with tf.variable_scope(self.name):
            state_inp = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
            action_input = tf.placeholder(tf.float32, [None, self.act_dim], name='action_input')
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', (self.state_dim, l1_size))
                b1 = tf.Variable(tf.zeros([l1_size,]), name='b')
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', (l1_size, l2_size))
                w2_act = tf.get_variable('w2_act', (self.act_dim, l2_size))
                b2 = tf.Variable(tf.zeros([l2_size,]), name='b')
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', (l2_size, 1))
                b3 = tf.Variable(tf.zeros([1,]), name='b')

            l1 = tf.nn.relu(tf.matmul(state_inp, w1) + b1)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + tf.matmul(action_input, w2_act) + b2)
            q_value = tf.identity(tf.matmul(l2, w3) + b3, name='q_value')

            action_gradients = tf.gradients(q_value, action_input,name='action_grads')
        net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return state_inp, action_input, q_value, action_gradients, net

    def create_updater(self):
        q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')
        loss = tf.reduce_mean(tf.square(q_target - self.q_value),name='critic_loss')
        update_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return q_target, loss, update_step

    def update(self, reward, states, actions):
        self.sess.run(self.update_step, feed_dict={self.q_target: reward,
                                                   self.state_input: states,
                                                   self.action_input: actions})

    def get_q(self, states, actions):
        return self.sess.run(self.q_value, feed_dict={self.state_input:states,
                                                      self.action_input:actions})

    def get_act_grads(self, states, actions):
        return self.sess.run(self.action_grads, feed_dice={self.state_input:states,
                                                           self.action_input:actions})


class DDPG:
    def __init__(self, max_buffer_len, batch_size, action_dim=60, state_dim=32,
                 start_train_at=100, test_verbose=10, save_model_every=25,
                 savedir='./DDPG/model/', logdir='./DDPG/logs', load=False, name='ddpg'):
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
                           act_dim=action_dim, l1_s=64, l2_s=128)
        self.critic = Critic(self.sess, 'critic', state_dim=state_dim,
                             act_dim=action_dim, l1_size=64, l2_size=64)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name[:-2], var)

        self.__summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(logdir)
        self.summary_writer.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if load:
            self.load_model()
            # TODO: somehow get the global_step from checkpoint

        self.buffer = ReplayBuffer(max_buffer_len)
        self.noize = OUNoise(action_dim)

    def train(self):
        train_batch = np.array(self.buffer.sample(self.bs))
        states = train_batch[:, 0]
        actions = train_batch[:, 1]
        rewards = train_batch[:, 2]

        self.critic.update(rewards, states, actions)

        actions_new = self.actor.get_actions(states)
        q_grads = self.critic.get_act_grads(states, actions_new)
        self.actor.apply_grads(q_grads, states)

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
            self.write_summary()
        if self.gs % self.save_model_every == self.save_model_every - 1:
            self.save_model()

    def save_model(self):
        self.saver.save(self.sess, self.savedir)

    def load_model(self):
        self.saver.restore(self.sess, self.savedir)

    def write_summary(self):
        s = self.sess.run(self.__summary)
        self.summary_writer.add_summary(s, self.gs)



if __name__ == '__main__':
    agent = DDPG(100, 16, 40)
    agent.write_summary()
    print(len(tf.trainable_variables()))
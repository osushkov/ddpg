
import agent
import memory
import actor_network
import critic_network
import constants

import math
import random
import numpy as np
import sonnet as snt
import tensorflow as tf

from gym.spaces.discrete import Discrete


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPGAgent(agent.Agent):

    def __init__(self, action_space, observation_space, exploration_rate, memory):
        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate
        self._actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._action_space.shape[0]))

        self._state_shape = observation_space.high.shape
        self._memory = memory
        self._cur_exploration = self._exploration_rate(0)

        self._last_action = None
        self._last_state = None

        self._build_graph()

        self._sess = tf.Session(graph=self._graph)
        with self._sess.as_default():
            self._sess.run(self._init_op)
            self._sess.run(self._actor.init_ops)
            self._sess.run(self._critic.init_ops)

    def initialize_episode(self, episode_count):
        self._cur_exploration = self._exploration_rate(episode_count)

    def act(self, observation):
        observation = self._normalised_state(observation)

        self._learn()

        observation = observation.reshape((1,) + self._state_shape)
        feed_dict = {
            self._act_noise: self._cur_exploration,
            self._act_observation: observation,
        }

        with self._sess.as_default():
            action = self._sess.run(self._act_output, feed_dict=feed_dict) + self._actor_noise()

        self._last_state = observation
        self._last_action = action

        return action

    def feedback(self, resulting_state, reward, episode_done):
        resulting_state = resulting_state.reshape((1,) + self._state_shape)
        resulting_state = self._normalised_state(resulting_state)

        if episode_done:
            resulting_state = None

        self._memory.add_memory(self._last_state, self._last_action, reward, resulting_state)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _learn(self):
        if self._memory.num_entries() < constants.LEARN_BATCH_SIZE:
            return

        mem_chunk = self._memory.sample(constants.LEARN_BATCH_SIZE)
        feed_dict = {
                self._state : mem_chunk.states,
                self._action : mem_chunk.actions.reshape(-1, self._action_space.shape[0]),
                self._reward : mem_chunk.rewards,
                self._next_state : mem_chunk.next_states,
                self._target_is_terminal : mem_chunk.is_terminal,
        }

        with self._sess.as_default():
            self._sess.run(self._critic_optimizer, feed_dict=feed_dict)
            self._sess.run(self._actor_optimizer, feed_dict=feed_dict)

            self._sess.run(self._actor.update_ops)
            self._sess.run(self._critic.update_ops)

    def _build_graph(self):
        self._graph = tf.Graph()

        action_ranges =  (self._action_space.low, self._action_space.high)

        with self._graph.as_default():
            self._actor = actor_network.Actor(constants.ACTOR_LAYERS,
                                              action_ranges,
                                              constants.LAYER_ACTIVATION)

            self._critic = critic_network.Critic(constants.CRITIC_LAYERS,
                                                 constants.LAYER_ACTIVATION)

            self._state = tf.placeholder(
                    tf.float32, shape=((constants.LEARN_BATCH_SIZE, ) + self._state_shape))
            self._action =  tf.placeholder(
                    tf.float32, shape=(constants.LEARN_BATCH_SIZE, self._action_space.shape[0]))
            self._reward = tf.placeholder(tf.float32, shape=constants.LEARN_BATCH_SIZE)
            self._next_state = tf.placeholder(
                    tf.float32, shape=((constants.LEARN_BATCH_SIZE, ) + self._state_shape))
            self._target_is_terminal = tf.placeholder(tf.bool, shape=constants.LEARN_BATCH_SIZE)

            self._build_acting_network()
            self._build_actor_learning_network()
            self._build_critic_learning_network()

            self._init_op = tf.global_variables_initializer()
            self._actor.after_connect_init()
            self._critic.after_connect_init()

    def _build_acting_network(self):
        self._act_observation = tf.placeholder(tf.float32, shape=((1, ) + self._state_shape))
        self._act_noise = tf.placeholder(tf.float32)

        self._act_output = self._actor.learner(self._act_observation)
                            # tf.random_normal(shape=(1,), stddev=self._act_noise))

    def _build_actor_learning_network(self):
        action = self._actor.learner(self._state)

        # TODO: should this be the critic or target_critic?
        qvalue = self._critic.learner(self._state, action)
        self._actor_loss = -tf.reduce_mean(qvalue)

        opt = tf.train.AdamOptimizer(0.0001)
        self._actor_optimizer = opt.minimize(self._actor_loss,
                                             var_list=self._actor.learner.get_variables())

    def _build_critic_learning_network(self):
        critic_output = tf.reshape(self._critic.learner(self._state, self._action), [-1])
        self._critic_output = critic_output

        next_state_action = self._actor.target(self._next_state)
        next_state_qvalue = tf.reshape(self._critic.target(self._next_state, next_state_action), [-1])

        terminating_target = self._reward
        intermediate_target = self._reward + next_state_qvalue * constants.DISCOUNT
        desired_output = tf.stop_gradient(
            tf.where(self._target_is_terminal, terminating_target, intermediate_target))

        self._td_error = desired_output - critic_output
        self._critic_loss = tf.losses.mean_squared_error(desired_output, critic_output)

        opt = tf.train.AdamOptimizer(0.001)
        self._critic_optimizer = opt.minimize(self._critic_loss,
                                              var_list=self._critic.learner.get_variables())

    def _normalised_state(self, obs):
        # obs[0] /= self._observation_space.high[0] / 2.0
        # obs[1] /= 1.5
        # obs[2] /= self._observation_space.high[2] / 2.0
        # obs[3] /= 1.5
        return obs
        # obs_range = (self._observation_space.high - self._observation_space.low)
        # return (obs - self._observation_space.low) / obs_range

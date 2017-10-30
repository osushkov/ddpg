
import agent
import memory
import q_network

import math
import random
import numpy as np
import tensorflow as tf

from gym.spaces.discrete import Discrete

_ACTOR_LAYERS = (32,16)
_ACTOR_ACTIVATION = tf.nn.elu

_CRITIC_LAYERS = (32,16)
_CRITIC_LAYER_ACTIVATION = tf.nn.elu
_CRITIC_OUTPUT_ACTIVATION = tf.identity

_TARGET_UPDATE_RATE = 5000
_LEARN_BATCH_SIZE = 64
_DISCOUNT = 0.98


class DDPGAgent(agent.Agent):

    def __init__(self, action_space, observation_space, exploration_rate, memory):
        assert isinstance(action_space, Discrete)

        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate

        self._state_shape = observation_space.high.shape
        self._memory = memory
        self._cur_exploration = self._exploration_rate(0)

        self._last_action = None
        self._last_state = None

        self._learn_iters_since_update = 0

        self._build_graph()

        self._sess = tf.Session(graph=self._graph)
        with self._sess.as_default():
            self._sess.run(self._init_op)
            self._sess.run(self._target_update_ops)

    def initialize_episode(self, episode_count):
        self._cur_exploration = self._exploration_rate(episode_count)
        self._memory.initialize_episode(episode_count)

    def act(self, observation):
        observation = self._normalised_state(observation)

        self._learn()

        observation = observation.reshape((1,) + self._state_shape)
        feed_dict = {
            self._act_noise: self._cur_exploration,
            self._act_observation: observation,
        }

        with self._sess.as_default():
            action = self._sess.run(self._act_output, feed_dict=feed_dict)

        self._last_state = observation
        self._last_action = action

        return action

    def feedback(self, resulting_state, reward, episode_done):
        resulting_state = self._normalised_state(resulting_state)
        reward /= 50.0

        if episode_done:
            resulting_state = None

        self._memory.add_memory(self._last_state, self._last_action, reward, resulting_state)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _learn(self):
        if self._memory.num_entries() < self._memory.capacity() / 10:
            return

        mem_chunk = self._memory.sample(_LEARN_BATCH_SIZE)
        feed_dict = {
                self._weights : mem_chunk.weights,
                self._state : mem_chunk.states,
                self._action : mem_chunk.actions,
                self._reward : mem_chunk.rewards,
                self._next_state : mem_chunk.next_states,
                self._target_is_terminal : mem_chunk.is_terminal,
        }

        self._learn_iters_since_update += 1
        with self._sess.as_default():
            _, _, td_error = self._sess.run((self._actor_optimizer, self._critic_optimizer,
                                                self._td_error), feed_dict=feed_dict)

            self._memory.update_p_choice(td_error)

            if self._learn_iters_since_update >= _TARGET_UPDATE_RATE:
                self._sess.run(self._target_update_ops)
                self._learn_iters_since_update = 0

    def _build_graph(self):
        self._graph = tf.Graph()

        action_ranges =  (self._action_space.low, self._action_space.high)

        with self._graph.as_default():
            self._actor = actor_network.ActorNetwork(_ACTOR_LAYERS,
                                                     action_ranges,
                                                     _ACTOR_ACTIVATION)
            self._target_actor = actor_network.ActorNetwork(_ACTOR_LAYERS,
                                                            action_ranges,
                                                            _ACTOR_ACTIVATION)

            self._critic = actor_network.ActorNetwork(_CRITIC_LAYERS,
                                                      _CRITIC_LAYER_ACTIVATION,
                                                      _CRITIC_OUTPUT_ACTIVATION)
            self._target_critic = actor_network.ActorNetwork(_CRITIC_LAYERS,
                                                             _CRITIC_LAYER_ACTIVATION,
                                                             _CRITIC_OUTPUT_ACTIVATION)

            self._state = tf.placeholder(
                    tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
            self._action =  tf.placeholder(
                    tf.float32, shape=(_LEARN_BATCH_SIZE, self._action_space.n))
            self._reward = tf.placeholder(tf.float32, shape=_LEARN_BATCH_SIZE)
            self._next_state = tf.placeholder(
                    tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
            self._target_is_terminal = tf.placeholder(tf.bool, shape=_LEARN_BATCH_SIZE)

            self._weights = tf.placeholder(tf.float32, shape=_LEARN_BATCH_SIZE)
            self._normalized_weights = tf.nn.l2_normalize(self._weights, 0)

            self._build_acting_network()
            self._build_actor_learning_network()
            self._build_critic_learning_network()

            self._build_target_network()
            self._build_learn_loss()
            self._build_update_ops()

            self._init_op = tf.global_variables_initializer()

    def _build_acting_network(self):
        self._act_observation = tf.placeholder(tf.float32, shape=((1, ) + self._state_shape))
        self._act_noise = tf.placeholder(tf.float32, shape=(1,))

        self._act_output = (self._actor(self._act_observation) +
                            tf.random_normal(shape=(1,), stddev=self._act_noise))

    def _build_actor_learning_network(self):
        action = self._actor(self._state)

        # TODO: should this be the critic or target_critic?
        qvalue = self._critic(self._state, action)
        self._actor_loss = -tf.log(qvalue) * self._normalized_weights

        opt = tf.train.AdamOptimizer(0.0001)
        self._actor_optimizer = opt.minimize(self._actor_loss,
                                             var_list=self._actor.get_variables())

    def _build_critic_learning_network(self):
        critic_output = self._critic(self._state, self._action)

        next_state_action = self._target_actor(self._next_state)
        next_state_qvalue = self._target_critic(self._next_state, next_state_action)

        terminating_target = self._reward
        intermediate_target = self._reward + next_state_qvalue * _DISCOUNT)
        desired_output = tf.stop_gradient(
            tf.where(self._target_is_terminal, terminating_target, intermediate_target))

        self._td_error = desired_output - critic_output
        self._critic_loss = tf.losses.mean_squared_error(desired_output, critic_output,
                                                         weights=self._normalized_weights)

        opt = tf.train.AdamOptimizer(0.0001)
        self._critic_optimizer = opt.minimize(self._critic_loss,
                                              var_list=self._critic.get_variables())

    def _build_update_ops(self):
        actor_vars = self._actor.get_variables()
        actor_target_vars = self._target_actor.get_variables()

        self._target_update_ops = []
        for src_var, dst_var in zip(actor_vars, actor_target_vars):
            self._target_update_ops.append(
                    tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

        critic_vars = self._critic.get_variables()
        critic_target_vars = self._target_critic.get_variables()

        for src_var, dst_var in zip(critic_vars, critic_target_vars):
            self._target_update_ops.append(
                    tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

    def _normalised_state(self, obs):
        # obs[0] /= self._observation_space.high[0] / 2.0
        # obs[1] /= 1.5
        # obs[2] /= self._observation_space.high[2] / 2.0
        # obs[3] /= 1.5
        return obs
        # obs_range = (self._observation_space.high - self._observation_space.low)
        # return (obs - self._observation_space.low) / obs_range

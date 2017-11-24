import sonnet as snt
import tensorflow as tf

import constants


class Critic(object):

    def __init__(self, layer_sizes, layer_activations):
        self._learner = CriticNetwork(layer_sizes, layer_activations)
        self._target = CriticNetwork(layer_sizes, layer_activations)

    def after_connect_init(self):
        self._init_ops = self._build_init_ops()
        self._update_ops = self._build_update_ops()

    def _build_init_ops(self):
        learner_vars = self._learner.get_variables()
        target_vars = self._target.get_variables()

        ops = []
        for src_var, dst_var in zip(learner_vars, target_vars):
            ops.append(tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))
        return ops

    def _build_update_ops(self):
        learner_vars = self._learner.get_variables()
        target_vars = self._target.get_variables()

        ops = []
        for src_var, dst_var in zip(learner_vars, target_vars):
            new_var = tf.multiply(src_var, constants.UPDATE_TAU) + tf.multiply(dst_var, 1.0 - constants.UPDATE_TAU)
            ops.append(dst_var.assign(new_var))
        return ops

    @property
    def learner(self):
        return self._learner

    @property
    def target(self):
        return self._target

    @property
    def init_ops(self):
        return self._init_ops

    @property
    def update_ops(self):
        return self._update_ops


class CriticNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, layer_activations):
        super(CriticNetwork, self).__init__(name='critic_network')

        regularizers = {
                'w' : tf.contrib.layers.l1_regularizer(scale=constants.CRITIC_REGULARIZATION),
                'b' : tf.contrib.layers.l2_regularizer(scale=constants.CRITIC_REGULARIZATION)
        }

        self._network = snt.nets.MLP(layer_sizes,
                                     activation=layer_activations,
                                     regularizers=regularizers)
        self._value = snt.Linear(output_size=1)

    def _build(self, state, action):
        state_flattened = snt.BatchFlatten()(state)
        action_flattened = snt.BatchFlatten()(action)
        state_action = tf.concat([state_flattened, action_flattened], 1)

        net_out = self._network(state_action)
        value = self._value(net_out)

        return value

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        return (self._network.get_variables(collection) +
                self._value.get_variables(collection))

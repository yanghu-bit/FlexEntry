from __future__ import print_function

import os
import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from rl_game import FlexEntry_Game
from utils import softmax
from model import Network
from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')

def sim(network, game):
    for tm_idx in game.tm_indexes:
        state = game.get_state(tm_idx)
        policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]
        game.evaluate(tm_idx, actions, eval_delay=False)


def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = FlexEntry_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim, mask=game.adjacent_matrix)

    step = network.restore_ckpt(FLAGS.ckpt)
    learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
  
    sim(network, game)


if __name__ == '__main__':
    app.run(main)

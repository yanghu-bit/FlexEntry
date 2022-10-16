from __future__ import print_function

import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from rl_game import FlexEntry_Game
from model import Network, Network_model
from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')

def sim(config, network, game, moves_queue):
    moves = []
    max_move_num = []
    max_move_num.append(0)

    model_action_dim = game.num_pairs
    model_1 = None
    model_2 = None
    model_3 = None
    model_4 = None
    model_5 = None
    model_6 = None
    if config.models_num >= 2:
        model_move_1 = moves_queue[1]
        model_move_2 = moves_queue[2]
        model_1 = Network_model(config, game.state_dims, model_action_dim, model_move_1, master=False)
        model_2 = Network_model(config, game.state_dims, model_action_dim, model_move_2, master=False)
        model_1.restore_ckpt(0, FLAGS.ckpt)
        model_2.restore_ckpt(1, FLAGS.ckpt)
        max_move_num.append(moves_queue[1])
        max_move_num.append(moves_queue[2])

    if config.models_num >= 3:
        model_move_3 = moves_queue[3]
        model_3 = Network_model(config, game.state_dims, model_action_dim, model_move_3, master=False)
        model_3.restore_ckpt(2, FLAGS.ckpt)
        max_move_num.append(moves_queue[3])

    if config.models_num >= 4:
        model_move_4 = moves_queue[4]
        model_4 = Network_model(config, game.state_dims, model_action_dim, model_move_4, master=False)
        model_4.restore_ckpt(3, FLAGS.ckpt)
        max_move_num.append(moves_queue[4])

    if config.models_num >= 5:
        model_move_5 = moves_queue[5]
        model_5 = Network_model(config, game.state_dims, model_action_dim, model_move_5, master=False)
        model_5.restore_ckpt(4, FLAGS.ckpt)
        max_move_num.append(moves_queue[5])

    if config.models_num >= 6:
        model_move_6 = moves_queue[6]
        model_6 = Network_model(config, game.state_dims, model_action_dim, model_move_6, master=False)
        model_6.restore_ckpt(5, FLAGS.ckpt)
        max_move_num.append(moves_queue[6])

    tm_count = 0

    for tm_idx in game.tm_indexes:
        tm_count += 1
        state = game.get_state(tm_idx)
        policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-1:]

        moves.append(max_move_num[actions[0]])

        if actions[0] == 0:
            actionsx = [-1]
        if actions[0] == 1:
            policy = model_1.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]
        if actions[0] == 2:
            policy = model_2.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]
        if actions[0] == 3:
            policy = model_3.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]
        if actions[0] == 4:
            policy = model_4.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]
        if actions[0] == 5:
            policy = model_5.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]
        if actions[0] == 6:
            policy = model_6.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[-max_move_num[actions[0]]:]

        game.evaluate(tm_idx, actionsx, eval_delay=False)

    file_handle = open('result.txt', mode='a')
    file_handle.write('moves:\n')
    file_handle.write(str(moves))
    file_handle.close()


def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = FlexEntry_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim)

    step = network.restore_ckpt(FLAGS.ckpt)
    learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n'% (step, learning_rate))

    moves_queue = []
    if config.model_interval == 'custom':
        moves_queue = config.model_critical_entries_number
  
    sim(config, network, game, moves_queue)


if __name__ == '__main__':
    app.run(main)

from __future__ import print_function

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import math
import datetime
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from rl_game import FlexEntry_Game
from utils import softmax
from model import Network, Network_model
from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean('cpu_only', True, 'using cpu for training')
flags.DEFINE_integer('num_agents', 20, 'number of agents')        ##changed agent
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
flags.DEFINE_integer('num_iter', 50, 'Number of iterations each agent would run')

flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')

GRADIENTS_CHECK=True

FREQ_EPSILON = 0.01

def central_agent(config, game, model_weights_queues, experience_queues, moves_queue):
    network = Network(config, game.state_dims, game.action_dim, mask=game.adjacent_matrix, master=True)
    network.save_hyperparams(config)
    start_step = network.restore_ckpt()
    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):
        network.ckpt.step.assign_add(1)
        model_weights = network.model.get_weights()

        for i in range(config.num_agents):
            model_weights_queues[i].put(model_weights)

        if config.method == 'actor_critic':
            #assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(config.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()
              
                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
           
            assert len(s_batch)*config.max_moves == len(a_batch)
            #used shared RMSProp, i.e., shared g
            """value_loss, entropy, actor_gradients, critic_gradients = network.actor_critic_train(np.array(s_batch), 
                                                                        np.array(a_batch), 
                                                                        np.array(r_batch).astype(np.float32), 
                                                                        config.entropy_weight)"""
            ##special changed
            count = 0
            for i in a_batch:
                if i != 0:
                    count = count + moves_queue[i]

            avg_moves = float(count)/float(len(a_batch))
            # print('count:', count)
            # print('num:', len(a_batch))
            # print('avg_move:', avg_moves)

            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            value_loss, entropy, actor_gradients, critic_gradients = network.actor_critic_train(np.array(s_batch), 
                                                                    actions, 
                                                                    np.array(r_batch).astype(np.float32), 
                                                                    config.entropy_weight)
       
            if GRADIENTS_CHECK:
                for g in range(len(actor_gradients)):
                    assert np.any(np.isnan(actor_gradients[g])) == False, ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in range(len(critic_gradients)):
                    assert np.any(np.isnan(critic_gradients[g])) == False, ('critic_gradients', s_batch, a_batch, r_batch)

            if step % config.save_step == config.save_step - 1:
                network.save_ckpt(_print=True)
                
                #log training information
                actor_learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy)

                network.inject_summaries({
                    'learning rate': actor_learning_rate,
                    'value loss': avg_value_loss,
                    'avg reward': avg_reward,
                    'avg entropy': avg_entropy,
                    'avg moves': np.float32(avg_moves)
                }, step)

                print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f'%(actor_learning_rate, avg_value_loss, avg_reward, avg_entropy))

def agent(agent_id, config, game, tm_subset, model_weights_queue, experience_queue, moves_queue):
    random_state = np.random.RandomState(seed=agent_id)
    network = Network(config, game.state_dims, game.action_dim, mask=game.adjacent_matrix, master=False)

    # initial synchronization of the model weights from the coordinator 
    model_weights = model_weights_queue.get()
    network.model.set_weights(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []

    run_iteration_idx = 0
    num_tms = len(tm_subset)

    run_iterations = FLAGS.num_iter

    #model number
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
        model_1 = Network_model(config, game.state_dims, model_action_dim, model_move_1, mask=game.adjacent_matrix, master=False)
        model_2 = Network_model(config, game.state_dims, model_action_dim, model_move_2, mask=game.adjacent_matrix, master=False)
        model_1.restore_ckpt(0, FLAGS.ckpt)
        model_2.restore_ckpt(1, FLAGS.ckpt)

    if config.models_num >= 3:
        model_move_3 = moves_queue[3]
        model_3 = Network_model(config, game.state_dims, model_action_dim, model_move_3, mask=game.adjacent_matrix, master=False)
        model_3.restore_ckpt(2, FLAGS.ckpt)

    if config.models_num >= 4:
        model_move_4 = moves_queue[4]
        model_4 = Network_model(config, game.state_dims, model_action_dim, model_move_4, mask=game.adjacent_matrix, master=False)
        model_4.restore_ckpt(3, FLAGS.ckpt)

    if config.models_num >= 5:
        model_move_5 = moves_queue[5]
        model_5 = Network_model(config, game.state_dims, model_action_dim, model_move_5, mask=game.adjacent_matrix, master=False)
        model_5.restore_ckpt(4, FLAGS.ckpt)

    if config.models_num >= 6:
        model_move_6 = moves_queue[6]
        model_6 = Network_model(config, game.state_dims, model_action_dim, model_move_6, mask=game.adjacent_matrix, master=False)
        model_6.restore_ckpt(5, FLAGS.ckpt)


    while True:
        if config.priority_sampling:
            sample_probability = softmax(sample_weight)
            idx = random_state.choice(num_tms, p=sample_probability)
        tm_idx = tm_subset[idx]
        #state
        state = game.get_state(tm_idx)
        s_batch.append(state)
        #action
        if config.method == 'actor_critic':    
            policy = network.actor_predict(np.expand_dims(state, 0), game.softmax_temperature).numpy()[0]
        assert np.count_nonzero(policy) >= config.max_moves, (policy, state)
        actions = random_state.choice(game.action_dim, config.max_moves, p=policy, replace=False)
        for a in actions:
            a_batch.append(a)

        #reward
        reward = game.reward(tm_idx, actions, moves_queue, model_1, model_2, model_3, model_4, model_5, model_6)
        r_batch.append(reward)
        if config.priority_sampling:
            sample_weight += FREQ_EPSILON
            sample_weight[idx] = (1/reward)*100

        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            # Report experience to the coordinator                          
            # instead of reporting gradients to the coordiantor
            if config.method == 'actor_critic':    
                experience_queue.put([s_batch, a_batch, r_batch])
            
            #print('report', agent_id)

            # synchronize the network parameters from the coordinator
            model_weights = model_weights_queue.get()
            network.model.set_weights(model_weights)
            
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            run_iteration_idx = 0
      
        if config.priority_sampling == False:
            # Update idx
            idx += 1

            if idx == num_tms:
               idx = 0


def main(_):
    if FLAGS.cpu_only:
        tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')
    #tf.debugging.set_log_device_placement(True)

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=True)
    game = FlexEntry_Game(config, env)
    model_weights_queues = []
    experience_queues = []
    if FLAGS.num_agents == 0:
        config.num_agents = mp.cpu_count() - 1
    #FLAGS.num_iter = env.tm_cnt//config.num_agents
    print('agent num: %d, iter num: %d\n'%(config.num_agents, FLAGS.num_iter))
    for _ in range(config.num_agents):
        model_weights_queues.append(mp.Queue(1))
        experience_queues.append(mp.Queue(1))

    tm_subsets = np.array_split(game.tm_indexes, config.num_agents)

    moves_queue = []
    if config.model_interval == 'custom':
        moves_queue = config.model_critical_entries_number

    coordinator = mp.Process(target=central_agent, args=(config, game, model_weights_queues, experience_queues, moves_queue))

    coordinator.start()

    tf.config.experimental.set_visible_devices([], 'GPU')
    agents = []
    for i in range(config.num_agents):
        agents.append(mp.Process(target=agent, args=(i, config, game, tm_subsets[i], model_weights_queues[i], experience_queues[i], moves_queue)))

    for i in range(config.num_agents):
        agents[i].start()

    coordinator.join()

if __name__ == '__main__':
    app.run(main)

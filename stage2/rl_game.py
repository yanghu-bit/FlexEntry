from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import numpy as np
from gurobipy import *
from utils import multi_processing
from game import Game

class FlexEntry_Game(Game):
    def __init__(self, config, env, random_seed=1000):
        super(FlexEntry_Game, self).__init__(config, env, random_seed)
        self.project_name = config.project_name
        self.max_moves = config.max_moves
        self.action_dim = config.models_num + 1
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        if config.softmax_temperature:
            self.softmax_temperature = np.sqrt(self.action_dim)
        else:
            self.softmax_temperature = 1

        self.tm_indexes = np.arange(0, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        self.generate_traffic_matrices(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        self.get_optimal_routing_mlu()

    def get_state(self, tm_idx):
        return self.normalized_traffic_matrices[tm_idx]

    def reward(self, tm_idx, actions, moves_queue, model_1, model_2, model_3, model_4, model_5, model_6):
        state = self.get_state(tm_idx)
        optimal_mlu = self.optimal_mlu_per_tm[tm_idx]
        if actions[0] == 0:
            ecmp_mlu, _ = self.eval_ecmp_traffic_distribution(tm_idx)
            reward = optimal_mlu / ecmp_mlu
            reward = reward[0]
        elif actions[0] == 1:
            policy = model_1.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[1]):]
        elif actions[0] == 2:
            policy = model_2.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[2]):]
        elif actions[0] == 3:
            policy = model_3.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[3]):]
        elif actions[0] == 4:
            policy = model_4.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[4]):]
        elif actions[0] == 5:
            policy = model_5.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[5]):]
        elif actions[0] == 6:
            policy = model_6.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            actionsx = policy.argsort()[(-1 * moves_queue[6]):]

        if actions[0] > 0:
            mlu, _ = self.optimal_routing_mlu_critical_entries_v2(tm_idx, actionsx)
            reward = optimal_mlu / mlu

        #In the paper, alpha is penalty factor: lambda
        alpha = 0.05

        cost_tuple = 0
        if actions[0] > 0:
            cost_tuple = float(moves_queue[actions[0]]) / float(self.num_pairs)

        if reward <= 0.9:
            reward = (reward) * (reward)
        else:
            reward = 1 - (alpha * cost_tuple)

        return reward

    def evaluate(self, tm_idx, actions=None, ecmp=True, weighted_ecmp=True, eval_delay=False, eval_entry_cnt=False, top_k=True):
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx)
        if weighted_ecmp:
            _, solution = self.optimal_routing_mlu_weighted_ecmp(tm_idx)
            wecmp_mlu, wecmp_delay = self.eval_weighted_traffic_distribution(tm_idx, solution, eval_delay=eval_delay)
        if top_k:
            top_k_actions = self.entries_top_k(tm_idx, len(actions))
            _, solution = self.optimal_routing_mlu_critical_entries_v2(tm_idx, top_k_actions)
            self.eliminate_loops(tm_idx, solution, top_k_actions)
            top_k_mlu, top_k_delay = self.eval_weighted_traffic_distribution(tm_idx, solution, eval_delay=eval_delay)
        #FlexEntry actions:
        if actions[0] == -1:
            mlu, delay = self.eval_ecmp_traffic_distribution(tm_idx)
            affected_ecmp_entry_cnt = 0
        else:
            _, solution = self.optimal_routing_mlu_critical_entries_v2(tm_idx, actions)
            affected_ecmp_entry_cnt = self.eliminate_loops(tm_idx, solution, actions)
            if affected_ecmp_entry_cnt == -100:
                mlu = [100]
                delay = [100]
            else:
                mlu, delay = self.eval_weighted_traffic_distribution(tm_idx, solution, eval_delay=eval_delay)

        #calculate and print result:
        try:
            assert eval_delay == False and eval_entry_cnt == False
            optimal_mlu = self.optimal_mlu_per_tm[tm_idx]
        except:
            _, solution = self.optimal_routing_mlu(tm_idx)
            optimal_mlu, optimal_mlu_delay, entry_cnt = self.eval_optimal_routing_mlu(tm_idx, solution, eval_entry_cnt=eval_entry_cnt)

        norm_mlu = optimal_mlu / mlu[0]
        line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu[0]) + ', ' + str(affected_ecmp_entry_cnt) + ', '

        if top_k:
            norm_top_k_mlu = optimal_mlu / top_k_mlu[t]
            line += str(norm_top_k_mlu) + ', ' + str(top_k_mlu[t]) + ', '
        if weighted_ecmp:
            norm_wecmp_mlu = optimal_mlu / wecmp_mlu[0]
            line += str(norm_wecmp_mlu) + ', ' + str(wecmp_mlu[0]) + ', '
        if ecmp:
            norm_ecmp_mlu = optimal_mlu / ecmp_mlu[0]
            line += str(norm_ecmp_mlu) + ', ' + str(ecmp_mlu[0]) + ', '

        if eval_entry_cnt:
            line += str(entry_cnt) + ', '

        if eval_delay:
            _, solution = self.optimal_routing_delay(tm_idx)
            optimal_delay, optimal_delay_mlu = self.eval_optimal_routing_delay(tm_idx, solution)

            line += str(optimal_delay/delay[0]) + ', '
            line += str(optimal_delay/optimal_mlu_delay) + ', '
            if preconfig_paths:
                line += str(optimal_delay/precfg_paths_mlu_delay[0]) + ', '
            if weighted_ecmp:
                line += str(optimal_delay/wecmp_delay[0]) + ', '
            if ecmp:
                line += str(optimal_delay/ecmp_delay[0]) + ', '

            assert tm_idx in self.load_multiplier, (tm_idx)
            line += str(self.load_multiplier[tm_idx]) + ', '

            print(line[:-2])
            file_handle = open('result.txt', mode='a')
            file_handle.write(line[:-2])
            file_handle.write('\n')
            file_handle.close()

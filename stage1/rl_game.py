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
        self.action_dim = env.num_pairs
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)

        self.softmax_temperature = 1
        
        self.tm_history = 1
        self.predict_interval = 0
        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        self.generate_traffic_matrices(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)

        self.normalized_reward = config.normalized_reward
        self.get_optimal_routing_mlu()

    def get_state(self, tm_idx):
        return self.normalized_traffic_matrices[tm_idx]

    def reward(self, tm_idx, actions):
        mlu, _ = self.optimal_routing_mlu_critical_entries_v2(tm_idx, actions)

        if self.normalized_reward:
            optimal_mlu = self.optimal_mlu_per_tm[tm_idx]
            reward = optimal_mlu / mlu
        else:
            reward = 1 / mlu

        return reward

    def evaluate(self, tm_idx, actions=None, ecmp=True, preconfig_paths=True, weighted_ecmp=True, eval_delay=False, eval_entry_cnt=False, eval_single=True):
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx)
        if weighted_ecmp:
            _, solution = self.optimal_routing_mlu_weighted_ecmp(tm_idx)
            wecmp_mlu, wecmp_delay = self.eval_weighted_traffic_distribution(tm_idx, solution, eval_delay=eval_delay)
        if preconfig_paths:
             _, solution = self.optimal_routing_mlu_preconfig_paths_v2(tm_idx) 
             precfg_paths_mlu, precfg_paths_mlu_delay = self.eval_preconfig_paths(tm_idx, solution, eval_delay=eval_delay)

        _, solution = self.optimal_routing_mlu_critical_entries_v2(tm_idx, actions)
        affected_ecmp_entry_cnt = self.eliminate_loops(tm_idx, solution, actions)
        #self.check_solution(tm_idx, solution)
        if affected_ecmp_entry_cnt == -100:
            mlu = [100]
            delay = [100]
        else:
            mlu, delay = self.eval_weighted_traffic_distribution(tm_idx, solution, eval_delay=eval_delay)

        try:
            assert eval_delay == False and eval_entry_cnt == False
            optimal_mlu = self.optimal_mlu_per_tm[tm_idx]
        except:
            _, solution = self.optimal_routing_mlu(tm_idx)
            optimal_mlu, optimal_mlu_delay, entry_cnt = self.eval_optimal_routing_mlu(tm_idx, solution, eval_entry_cnt=eval_entry_cnt)

        norm_mlu = optimal_mlu / mlu[0]
        line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu[0]) + ', ' + str(affected_ecmp_entry_cnt) + ', '

        if self.predict_interval > 0:
            norm_precfg_paths_mtms_mlu = optimal_mlu / precfg_paths_mtms_mlu[0]
            line += str(norm_precfg_paths_mtms_mlu) + ', ' + str(precfg_paths_mtms_mlu[0]) + ', '
        if preconfig_paths:
            norm_precfg_paths_mlu = optimal_mlu / precfg_paths_mlu[0]
            line += str(norm_precfg_paths_mlu) + ', ' + str(precfg_paths_mlu[0]) + ', '
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

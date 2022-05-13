from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
from tqdm import tqdm
import numpy as np
from gurobipy import *
from utils import multi_processing
#import matplotlib.pyplot as plt

OBJ_EPSILON = 1e-12
_NEG_INF_FP32 = -1e9

class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
        self.rebalance_percentage = 1
        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.max_paths_per_pair = env.max_paths_per_pair 
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.preconfig_paths_link = env.preconfig_paths_link            # paths with link info
        self.shortest_paths_node = env.shortest_paths_node              # paths with node info
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.neighbors = env.neighbors
        self.failure_links = None
        self.removed_pair_paths = None
        self.lp_solver = config.LP_solver
        self.paths = env.shortest_paths_link
        self.entries_load = {}

        self.get_ecmp_next_hops()
        
        self.model_type = config.model_type

        #for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.lp_link_capacity = {}
        for e in self.link_sd_to_idx:
            self.lp_link_capacity[e] = self.link_capacities[self.link_sd_to_idx[e]]
        self.num_paths_per_pair = [len(pr) for pr in self.preconfig_paths_link]    #[pair, num_paths]
        assert len(self.num_paths_per_pair) == self.num_pairs
        self.pair_paths = [(pr, ph) for pr in self.lp_pairs for ph in range(self.num_paths_per_pair[pr])]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
        self.outgoing_links = env.outgoing_links

        # attention mask
        self.adjacent_matrix = None

        #self.get_link_precfg_path_cnt()
        self.load_multiplier = {}

        self.get_possible_link_failure()

    def generate_traffic_matrices(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, 1]
        for tm_idx in self.tm_indexes:
            if normalization:
                tm_max_element = np.max(self.traffic_matrices[tm_idx])
                self.normalized_traffic_matrices[tm_idx,:,:,0] = self.traffic_matrices[tm_idx] / tm_max_element        #[Valid_tms, Node, Node, 1]
            else:
                self.normalized_traffic_matrices[tm_idx,:,:,0] = self.traffic_matrices[tm_idx]                         #[Valid_tms, Node, Node, 1]

    def get_possible_link_failure(self):
        self.possible_link_failure = []
        for link_idx in range(self.num_links):
            for pr in range(self.num_pairs):
                disconnected = True
                for ph in range(self.num_paths_per_pair[pr]):
                    if link_idx not in self.preconfig_paths_link[pr][ph]:
                        disconnected = False
                        break
                if disconnected:
                    #print('Invalid link failure: %d'%link_idx)
                    break
            if disconnected == False:
                self.possible_link_failure.append(link_idx)
        
        self.num_single_failures = len(self.possible_link_failure)
        #print('Possible link failure:', self.possible_link_failure)

    def load_optimal_routing_mlu(self, link_failures=None):
        if link_failures is None:
            self.optimal_mlu_file = self.traffic_file+'_optimal_mlu_per_tm'
        else:
            self.optimal_mlu_file = self.traffic_file+'_optimal_mlu_per_tm_'+str(link_failures)
        self.optimal_mlu_per_tm = []
        if os.path.exists(self.optimal_mlu_file):
            print('[*] Loading optimal mlu...', self.optimal_mlu_file)
            f = open(self.optimal_mlu_file, 'r')
            for line in f:
                item = line.strip()
                self.optimal_mlu_per_tm.append(float(item))
            f.close()
            if len(self.optimal_mlu_per_tm) == self.tm_cnt:
                return True
            else:
                print(len(self.optimal_mlu_per_tm), self.tm_cnt)

        return False

    def get_optimal_routing_mlu(self, link_failures=None):
        if self.load_optimal_routing_mlu(link_failures) == False:
            print('[!] Calculating optimal mlu...')
            func = []
            func.append(self.optimal_routing_mlu_dest_based)
            self.optimal_mlu_per_tm = multi_processing(func, np.arange(0, self.tm_cnt), ret_idx=0, ret_name='optimal mlu')
            assert len(self.optimal_mlu_per_tm) == self.tm_cnt, (len(self.optimal_mlu_per_tm), self.tm_cnt)
            print('[*] Saving optimal mlu...')
            f = open(self.optimal_mlu_file, 'w+')
            for tm_idx in range(self.tm_cnt):
                line = str(self.optimal_mlu_per_tm[tm_idx])
                f.writelines(line+'\n')
                print(tm_idx, self.optimal_mlu_per_tm[tm_idx])

            f.close()

    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    """def update_action_mask(self, action):
        
        assert self.mask[action] == 0

        self.mask[action] = _NEG_INF_FP32"""

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        #if next_hops_cnt > 1:
            #print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        #self.entries_load[(src, dst)] = self.entries_load.get((src, dst), 0) + demand
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        num_tms = 1
        eval_link_loads = np.zeros((num_tms, self.num_links))
        eval_max_utilization = np.zeros((num_tms))
        delay = np.zeros((num_tms))
        for t in range(num_tms):
            eval_link_loads[t] = self.ecmp_traffic_distribution(tm_idx+t)
            eval_max_utilization[t] = np.max(eval_link_loads[t] / self.link_capacities)
            self.load_multiplier[tm_idx+t] = 0.9 / eval_max_utilization[t]
            if eval_delay:
                eval_link_loads[t] *= self.load_multiplier[tm_idx+t]
                delay[t] = sum(eval_link_loads[t] / (self.link_capacities - eval_link_loads[t]))
            if self.failure_links is not None:
                for l in self.failure_links:
                    assert eval_link_loads[t][l] == 0, (t, l, eval_link_loads[t][l])

        return eval_max_utilization, delay

    def weighted_next_hop_distribution(self, link_loads, solution, demand, src, dst, visited_nodes):
        if src == dst:
            return

        next_hops = []
        ld_sum = 0
        for ld in solution:
            if solution[ld] > 1e-10 and ld[0] == src and ld[2] == dst:
                next_hops.append(ld[1])
                ld_sum += solution[ld]

        next_hops_cnt = len(next_hops)
        assert next_hops_cnt > 0 and ld_sum > 0, (demand, dst, visited_nodes)

        for np in next_hops:
            assert np not in visited_nodes, (src, np, dst, solution[src, np, dst], ld_sum)
            next_demand = demand * (solution[src, np, dst]/ld_sum)
            link_loads[self.link_sd_to_idx[(src, np)]] += next_demand
            self.weighted_next_hop_distribution(link_loads, solution, next_demand, np, dst, visited_nodes[:]+[np])

    def weighted_traffic_distribution(self, tm_idx, solution):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d or tm[s][d] == 0:
                    continue
                
                self.weighted_next_hop_distribution(link_loads, solution, tm[s][d], s, d, [s])
        
        return link_loads

    def traverse_next_hop(self, solution, src, dst, visited_nodes, affected_entries):
        if src == dst:
            return

        next_hops = []
        for ld in solution:
            if solution[ld] > 1e-10 and ld[0] == src and ld[2] == dst:
                next_hops.append(ld[1])

        next_hops_cnt = len(next_hops)
        #assert next_hops_cnt > 0, (dst, visited_nodes)
        if next_hops_cnt <= 0:
            return -100

        for np in next_hops:
            if np not in visited_nodes:
                temp = self.traverse_next_hop(solution, np, dst, visited_nodes[:]+[np], affected_entries)
                if temp == -100:
                    return -100
            else:
                # find a loop
                for n in range(len(visited_nodes)):
                    if visited_nodes[n] == np:
                        break
                nodes = visited_nodes[n:] + [np]
                node_cnt = len(nodes)
                # find the minimum link load on the loop
                min_link_load_dest = 1e10
                for i in range(node_cnt-1):
                    if solution[nodes[i], nodes[i+1], dst] < min_link_load_dest:
                        min_link_load_dest = solution[nodes[i], nodes[i+1], dst]
                # subtract the minimum link load from the loop, if it is non zero.
                if min_link_load_dest > 1e-10:
                    for i in range(node_cnt-1):
                        solution[nodes[i], nodes[i+1], dst] -= min_link_load_dest
                        if (nodes[i], dst) not in affected_entries:
                            affected_entries.append((nodes[i], dst))
                #print(nodes, min_link_load_dest, affected_entries)
        return 1

    def eliminate_loops(self, tm_idx, solution, actions):
        tm = self.traffic_matrices[tm_idx]
        affected_entries = []
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d or tm[s][d] == 0:
                    continue
                     
                temp = self.traverse_next_hop(solution, s, d, [s], affected_entries)
                if temp == -100:
                    return -100

        selected_node_dests = []
        for a in actions:
            selected_node_dests.append(self.pair_idx_to_sd[a])

        affected_ecmp_entries = []
        for e in affected_entries:
            if e not in selected_node_dests:
                affected_ecmp_entries.append(e)

        #print('Affected ecmp entries:', len(affected_ecmp_entries), affected_ecmp_entries)

        return len(affected_ecmp_entries)
            
    def output_preconfig_paths_split_ratios(self, tm_idx, pair_paths, solution):
        w = open(os.getcwd()+'/preconfig_paths_split_ratios', 'a+')
        w.writelines(str(tm_idx)+':\n') 
        line = ""
        current_pair = -1
        for pp in pair_paths:
            if current_pair != pp[0]:
                if line != "":
                    w.writelines(line+'\n')
                    #print(line+'\n')
                s, d = self.pair_idx_to_sd[pp[0]]
                line = str(s) + '->' + str(d) + ': ' + str(pp[1]) + '(' + str(solution[pp]) + ') '
                current_pair = pp[0]
            else:
                line += str(pp[1]) + '(' + str(solution[pp]) + ') '
       
        w.writelines('\n')
        w.close()

    def optimal_routing_mlu_dest_based(self, tm_idx=-1, demands=None, pairs=None, nodes=None, links=None):
        if pairs is None:
            pairs = self.lp_pairs
        pairs_sd = []
        for pr in pairs:
            pairs_sd.append(self.pair_idx_to_sd[pr])
        if demands is None:
            tm = self.traffic_matrices[tm_idx]
        else:
            tm = np.zeros((self.num_nodes, self.num_nodes))
            for pr in demands:
                assert pr in pairs
                s, d = self.pair_idx_to_sd[pr]
                tm[s][d] = demands[pr]
        if nodes is None:
            nodes = self.lp_nodes
        if links is None:
            links = self.links
        next_hops = [(n, i, d) for n, d in pairs_sd for _, i in self.outgoing_links[n] if (i, d) in pairs_sd or i == d]

        if self.lp_solver == 'Gurobi':
            m = Model('routing')
            m.Params.OutputFlag = 0

            next_hop_loads = m.addVars(next_hops, lb=0, name='next_hop_loads')
            link_load = m.addVars(links, name='link_load')
            r = m.addVar(name='congestion_ratio')

            m.addConstrs(
                (next_hop_loads.sum('*', i, d) - next_hop_loads.sum(i, '*', d) == -tm[i][d] for i, d in pairs_sd),
                "node_dest_constr1")
            # m.addConstrs((next_hop_loads.sum('*', d, d) - next_hop_loads.sum(d, '*', d) == sum(tm[ps][pd] for ps,pd in pairs_sd if pd==d) for d in nodes), "node_dest_constr2")
            m.addConstrs(
                (next_hop_loads.sum('*', d, d) == sum(tm[ps][pd] for ps, pd in pairs_sd if pd == d) for d in nodes),
                "node_dest_constr2")

            m.addConstrs(
                (link_load[e] == next_hop_loads.sum(self.link_idx_to_sd[e][0], self.link_idx_to_sd[e][1], '*') for e in
                 links), "link_load_constr")

            m.addConstrs((link_load[e] <= self.link_capacities[e] * r for e in links), "congestion_ratio_constr")

            # m.setObjective(r + OBJ_EPSILON*next_hop_loads.sum('*', '*', '*'), GRB.MINIMIZE)
            m.setObjective(r + OBJ_EPSILON * link_load.sum('*', '*', '*'), GRB.MINIMIZE)

            m.optimize()

            assert m.status == GRB.Status.OPTIMAL
            # print('Obj:', m.objVal)
            obj_r = r.X
            solution = m.getAttr('x', next_hop_loads)
        elif self.lp_solver == 'PuLP':
            model = LpProblem(name="routing")
            next_hop_loads = LpVariable.dicts(name="next_hop_loads", indexs=next_hops, lowBound=0)
            link_load = LpVariable.dicts(name="link_load", indexs=links)
            r = LpVariable(name="congestion_ratio")

            for i, d in pairs_sd:
                model += (
                lpSum([next_hop_loads[n, i, d] for n, _ in self.incoming_links[i] if (n, d) in pairs_sd]) - lpSum(
                    [next_hop_loads[i, n, d] for _, n in self.outgoing_links[i] if (n, d) in pairs_sd or n == d]) == -
                tm[i][d], "node_dest_constr%d_%d" % (i, d))

            for d in nodes:
                # model += (lpSum([next_hop_loads[n,d,d] for n,_ in self.incoming_links[d] if (n,d) in pairs_sd]) - lpSum([next_hop_loads[d,n,d] for _,n in self.outgoing_links[d] if (n,d) in pairs_sd]) == lpSum([tm[ps][pd] for ps,pd in pairs_sd if pd==d]), "node_dest_constr%d"%d)
                model += (
                lpSum([next_hop_loads[n, d, d] for n, _ in self.incoming_links[d] if (n, d) in pairs_sd]) == lpSum(
                    [tm[ps][pd] for ps, pd in pairs_sd if pd == d]), "node_dest_constr%d" % d)

            for e in links:
                es, ed = self.link_idx_to_sd[e]
                model += (link_load[e] == lpSum(
                    [next_hop_loads[es, ed, d] for d in nodes if es != d if (ed, d) in pairs_sd or ed == d]),
                          "link_load_constr%d" % e)
                model += (link_load[e] <= self.link_capacities[e] * r, "congestion_ratio_constr%d" % e)

            model += r + OBJ_EPSILON * lpSum([link_load[e] for e in links])

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in next_hop_loads:
                solution[k] = next_hop_loads[k].value()

        return obj_r, solution
       
    def eval_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False, eval_entry_cnt=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))
        if self.failure_links is not None:
            for l in self.failure_links:
                assert optimal_link_loads[l] == 0, (l, optimal_link_loads[l])

        # Calculate forwarding entries
        entry_cnt = 0
        if eval_entry_cnt:
            for i in range(self.num_pairs):
                for n in range(self.num_nodes):
                    install = 0
                    for e in self.lp_links:
                        if n == e[0] and solution[i, n, e[1]] > 1e-6:
                            install = 1
                            break
                    if install == 1:
                        entry_cnt += 1
                        
        return optimal_max_utilization, delay, entry_cnt
        
    def optimal_routing_mlu_weighted_ecmp(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]

        ecmp_nodes_dest = []
        for n in self.lp_nodes:
            for d in self.lp_nodes:
                if n != d:
                    ecmp_nodes_dest.append((n,d))

        links_dest = [(n, i, d) for n,d in ecmp_nodes_dest for i in self.ecmp_next_hops[n,d]]

        m = Model('routing')
        m.Params.OutputFlag = 0
        
        link_loads_dest = m.addVars(links_dest, name='link_loads_dest')
        
        r = m.addVar(name='congestion_ratio')

        m.addConstrs((link_loads_dest[ld] >= 0 for ld in links_dest), "link_load_dest_constr")

        m.addConstrs((link_loads_dest.sum('*', n, d) - link_loads_dest.sum(n, '*', d) == -tm[n][d] for n,d in ecmp_nodes_dest), "node_dest_constr1")
        
        m.addConstrs((link_loads_dest.sum('*', d, d) == sum(tm[s][d] for s in self.lp_nodes if s!=d) for d in self.lp_nodes), "node_dest_constr2")
        
        m.addConstrs((link_loads_dest.sum(e[0], e[1], '*') <= self.lp_link_capacity[e]*r for e in self.lp_links), "congestion_ratio_constr")
        
        m.setObjective(r, GRB.MINIMIZE)
        
        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        #print('Obj:', m.objVal)
        #print(m.getAttr('x', f))
        solution = m.getAttr('x', link_loads_dest)

        return r.X, solution
        
    def eval_weighted_traffic_distribution(self, tm_idx, solution, eval_delay=False):
        num_tms = 1
        eval_link_loads = np.zeros((num_tms, self.num_links))
        eval_max_utilization = np.zeros((num_tms))
        delay = np.zeros((num_tms))
        for t in range(num_tms):
            eval_link_loads[t] = self.weighted_traffic_distribution(tm_idx+t, solution)
            eval_max_utilization[t] = np.max(eval_link_loads[t] / self.link_capacities)
            if eval_delay:
                delay[t] = sum(eval_link_loads[t] / (self.link_capacities - eval_link_loads[t]))
        
        return eval_max_utilization, delay

    def optimal_routing_mlu_critical_entries_v2(self, tm_idx, actions):
        tm = self.traffic_matrices[tm_idx]

        nodes_dest = []
        for a in actions:
            nodes_dest.append(self.pair_idx_to_sd[a])

        ecmp_nodes_dest = []
        for n in self.lp_nodes:
            for d in self.lp_nodes:
                if n != d and (n,d) not in nodes_dest:
                    ecmp_nodes_dest.append((n,d))

        #links_dest = [(e[0], e[1], d) for e in self.lp_links for d in self.lp_nodes]
        links_dest = [(n, i, d) for n,d in nodes_dest for i in self.neighbors[n]]
        links_dest += [(n, i, d) for n,d in ecmp_nodes_dest for i in self.ecmp_next_hops[n,d]]

        m = Model('routing')
        m.Params.OutputFlag = 0
        
        link_loads_dest = m.addVars(links_dest, name='link_loads_dest')
        
        r = m.addVar(name='congestion_ratio')

        m.addConstrs((link_loads_dest[ld] >= 0 for ld in links_dest), "link_load_dest_constr")

        #m.addConstrs((link_loads_dest.sum('*', n, d) - link_loads_dest.sum(n, '*', d) == -tm[n][d] for n in self.lp_nodes for d in self.lp_nodes if n!=d ), "node_dest_constr1")
        m.addConstrs((link_loads_dest.sum('*', n, d) - link_loads_dest.sum(n, '*', d) == -tm[n][d] for n,d in nodes_dest), "node_dest_constr1")
        
        #m.addConstrs((link_loads_dest.sum('*', d, d) - link_loads_dest.sum(d, '*', d) == sum(tm[s][d] for s in self.lp_nodes if s!=d) for d in self.lp_nodes), "node_dest_constr2")
        m.addConstrs((link_loads_dest.sum('*', d, d) == sum(tm[s][d] for s in self.lp_nodes if s!=d) for d in self.lp_nodes), "node_dest_constr2")
        
        m.addConstrs(((link_loads_dest.sum('*', n, d) + tm[n][d])/len(self.ecmp_next_hops[n,d]) == link_loads_dest[n, i, d] for n,d in ecmp_nodes_dest for i in self.ecmp_next_hops[n,d]), "node_dest_constr3")
        
        m.addConstrs((link_loads_dest.sum(e[0], e[1], '*') <= self.lp_link_capacity[e]*r for e in self.lp_links), "congestion_ratio_constr")
        
        m.setObjective(r + OBJ_EPSILON*link_loads_dest.sum('*', '*', '*'), GRB.MINIMIZE)
        
        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        #print('Obj:', m.objVal)
        solution = m.getAttr('x', link_loads_dest)

        return r.X, solution

    def optimal_routing_mlu_preconfig_paths_v2(self, tm_idx, single_path=False, output_solution=False):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
        
        if self.lp_solver == 'Gurobi':
            m = Model('routing')
            m.Params.OutputFlag = 0
           
            if single_path == False:
                split_ratio = m.addVars(self.pair_paths, name='split_ratio')
            else:
                split_ratio = m.addVars(self.pair_paths, vtype=GRB.BINARY, name='split_ratio')
            
            link_load = m.addVars(self.lp_links, name='link_load')
            
            r = m.addVar(name='congestion_ratio')

            m.addConstrs((split_ratio[pp] >= 0 for pp in self.pair_paths), "split_ratio_constr1")
            m.addConstrs((split_ratio.sum(pr, '*') == 1 for pr in self.lp_pairs), "split_ratio_constr2")

            m.addConstrs((link_load[e] == sum(demands[pp[0]]*split_ratio[pp] for pp in self.pair_paths if self.link_sd_to_idx[e] in self.preconfig_paths_link[pp[0]][pp[1]]) for e in self.lp_links), "link_load_constr")
            
            m.addConstrs((link_load[e] <= self.lp_link_capacity[e]*r for e in self.lp_links), "congestion_ratio_constr")
           
            if single_path:
                m.setObjective(r, GRB.MINIMIZE)
            else:
                m.setObjective(r + OBJ_EPSILON*sum(bool(sr) for sr in split_ratio), GRB.MINIMIZE)
            
            m.optimize()

            assert m.status == GRB.Status.OPTIMAL
            #print('Obj:', m.objVal)
            #print(m.getAttr('x', f))
            obj_r = r.X
            solution = m.getAttr('x', split_ratio)
        elif self.lp_solver == 'PuLP':
            model = LpProblem(name="routing")

            if single_path == False:
                split_ratio = LpVariable.dicts(name="split_ratio", indexs=self.pair_paths, lowBound=0)
            else:
                split_ratio = LpVariable.dicts(name="split_ratio", indexs=self.pair_paths, lowBound=0, cat="Integer")
           
            link_load = LpVariable.dicts(name="link_load", indexs=self.links)

            r = LpVariable(name="congestion_ratio")
           
            for pr in self.lp_pairs:
                model += (lpSum([split_ratio[pr, ph] for _pr, ph in self.pair_paths if _pr == pr]) == 1, "split_ratio_constr%d"%pr)

            for e in self.links:
                model += (link_load[e] == lpSum([demands[pp[0]]*split_ratio[pp] for pp in self.pair_paths if e in self.preconfig_paths_link[pp[0]][pp[1]]]), "link_Load_constr%d"%e)

            for e in self.links:
                model += (link_load[e] <= self.link_capacities[e]*r, "congestion_ratio_constr%d"%e)

            if single_path == False:
                #model += r + OBJ_EPSILON*lpSum([split_ratio[pp] > 0 for pp in self.pair_paths])
                model += r
            else:
                model += r

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in split_ratio:
                solution[k] = split_ratio[k].value()

        if output_solution:
            self.output_preconfig_paths_split_ratios(self.pair_paths, solution)
        
        return obj_r, solution

    def eval_preconfig_paths(self, tm_idx, solution, eval_delay=False):
        eval_tms = self.traffic_matrices[tm_idx:tm_idx+1]
        num_tms = 1
        eval_link_loads = np.zeros((num_tms, self.num_links))
        eval_max_utilization = np.zeros((num_tms))
        delay = np.zeros((num_tms))
        for t in range(num_tms):
            eval_tm = eval_tms[t]
            for pp in self.pair_paths:
                #assert solution[pp] >= 0, (pp, solution[pp])
                if solution[pp] > 0:
                    s, d = self.pair_idx_to_sd[pp[0]]
                    eval_demand = eval_tm[s][d]*solution[pp]
                    if eval_demand > 0:
                        """for e in self.preconfig_paths_link[pp[0]][pp[1]]:
                            eval_link_loads[t][e] += eval_demand"""
                        eval_link_loads[t][self.preconfig_paths_link[pp[0]][pp[1]]] += eval_demand

            eval_max_utilization[t] = np.max(eval_link_loads[t] / self.link_capacities)
            if eval_delay:
                assert tm_idx+t in self.load_multiplier, (tm_idx+t)
                eval_link_loads[t] *= self.load_multiplier[tm_idx+t]
                delay[t] = sum(eval_link_loads[t] / (self.link_capacities - eval_link_loads[t]))
            if self.failure_links is not None:
                for l in self.failure_links:
                    assert eval_link_loads[t][l] == 0, (t, l, eval_link_loads[t][l])

        return eval_max_utilization, delay

    def optimal_routing_delay(self, tm_idx):
        assert tm_idx in self.load_multiplier, (tm_idx)
        tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        m = Model('routing')
        m.Params.OutputFlag = 0

        ratio = m.addVars(self.lp_pairs, self.lp_links, name='ratio')

        link_load = m.addVars(self.lp_links, name='link_load')

        f = m.addVars(self.lp_links, name='link_cost')
        
        m.addConstrs((ratio[pr, e[0], e[1]]>=0 for pr in self.lp_pairs for e in self.lp_links), "ratio_constr1")
        m.addConstrs((ratio[pr, e[0], e[1]]<=1 for pr in self.lp_pairs for e in self.lp_links), "ratio_constr2")
        m.addConstrs(
                (ratio.sum(pr, '*', self.pair_idx_to_sd[pr][0]) - ratio.sum(pr, self.pair_idx_to_sd[pr][0], '*') == -1 for pr in self.lp_pairs), 
                "flow_conservation_constr1")
        m.addConstrs(
                (ratio.sum(pr, '*', self.pair_idx_to_sd[pr][1]) - ratio.sum(pr, self.pair_idx_to_sd[pr][1], '*') == 1 for pr in self.lp_pairs), 
                "flow_conservation_constr2")
        m.addConstrs(
            (ratio.sum(pr, '*', i) - ratio.sum(pr, i, '*') == 0 for i in self.lp_nodes for pr in self.lp_pairs if i != self.pair_idx_to_sd[pr][0] and i != self.pair_idx_to_sd[pr][1]), "flow_conservation_constr3")

        m.addConstrs((link_load[e] == sum(demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs) for e in self.lp_links), "link_load_constr")

        m.addConstrs((f[e] >= link_load[e]/self.lp_link_capacity[e] for e in self.lp_links), "cost_constr1")
        m.addConstrs((f[e] >= 3*link_load[e]/self.lp_link_capacity[e] - 2/3. for e in self.lp_links), "cost_constr2")
        m.addConstrs((f[e] >= 10*link_load[e]/self.lp_link_capacity[e] - 16/3. for e in self.lp_links), "cost_constr3")
        m.addConstrs((f[e] >= 70*link_load[e]/self.lp_link_capacity[e] - 178/3. for e in self.lp_links), "cost_constr4")
        m.addConstrs((f[e] >= 500*link_load[e]/self.lp_link_capacity[e] - 1468/3. for e in self.lp_links), "cost_constr5")
        m.addConstrs((f[e] >= 5000*link_load[e]/self.lp_link_capacity[e] - 16318/3. for e in self.lp_links), "cost_constr6")
        
        m.setObjective(sum(f[e] for e in self.lp_links), GRB.MINIMIZE)

        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        #print('Obj:', m.objVal)
        solution = m.getAttr('x', ratio)
       
        return solution

    def eval_optimal_routing_delay(self, tm_idx, solution, eval_entry_cnt=False):
        optimal_link_loads = np.zeros((self.num_links))
        assert tm_idx in self.load_multiplier, (tm_idx)
        eval_tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))
        optimal_cost = 0
        for link_idx in range(self.num_links):
            optimal_cost += piecewise_function(optimal_link_loads[link_idx], self.link_capacities[link_idx])

        max_utilization = np.max(optimal_link_loads / self.link_capacities)
        
        # Calculate forwarding entries
        entry_cnt = 0
        if eval_entry_cnt:
            for i in range(self.num_pairs):
                for n in range(self.num_nodes):
                    install = 0
                    for e in self.lp_links:
                        if n == e[0] and solution[i, n, e[1]] > 1e-6:
                            install = 1
                            break
                    if install == 1:
                        entry_cnt += 1
                        
        #print('optimal_routing_delay:', optimal_delay, optimal_cost, m.objVal)

        return optimal_delay, max_utilization, entry_cnt

    def get_entries_loads(self, demand, src, dst):
        if src == dst:
            return
        ecmp_next_hops = self.ecmp_next_hops[src, dst]
        next_hops_cnt = len(ecmp_next_hops)
        ecmp_demand = demand / next_hops_cnt
        self.entries_load[(src, dst)] = self.entries_load.get((src, dst), 0) + demand
        for np in ecmp_next_hops:
            self.get_entries_loads(ecmp_demand, np, dst)

    def entries_top_k(self, tm_idx, length_actions):
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.get_entries_loads(demand, s, d)
        actions = []
        for i in range(length_actions):
            cur_k, cur_v = (-1, -1), -1
            for k, v in self.entries_load.items():
                if v > cur_v:
                    cur_v = v
                    cur_k = k
            del self.entries_load[cur_k]
            cur_action = self.pair_sd_to_idx[cur_k]
            actions.append(cur_action)

        return actions

def piecewise_function(link_load, link_capacity):
    utilization = link_load / link_capacity
    assert link_load >= 0 and link_capacity >= 0
    if utilization < 1/3.:
        return utilization
    elif utilization < 2/3.:
        return 3*utilization - 2/3.
    elif utilization < 9/10.:
        return 10*utilization - 16/3.
    elif utilization < 1.:
        return 70*utilization - 178/3.
    elif utilization < 11/10.:
        return 500*utilization - 1468/3.
    
    return 5000*utilization - 16318/3.


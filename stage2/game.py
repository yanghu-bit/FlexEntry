from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from gurobipy import *
from utils import multi_processing

OBJ_EPSILON = 1e-12

class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)

        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node              # paths with node info
        self.paths = env.shortest_paths_link                            # paths with link info
        self.neighbors = env.neighbors
        self.entries_load = {}

        self.get_ecmp_next_hops()

        #for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.lp_link_capacity = {}
        for e in self.link_sd_to_idx:
            self.lp_link_capacity[e] = self.link_capacities[self.link_sd_to_idx[e]]
        self.outgoing_links = env.outgoing_links

        self.load_multiplier = {}

    def generate_traffic_matrices(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, 1]
        for tm_idx in self.tm_indexes:
            if normalization:
                tm_max_element = np.max(self.traffic_matrices[tm_idx])
                self.normalized_traffic_matrices[tm_idx,:,:,0] = self.traffic_matrices[tm_idx] / tm_max_element        #[Valid_tms, Node, Node, 1]
            else:
                self.normalized_traffic_matrices[tm_idx,:,:,0] = self.traffic_matrices[tm_idx]                         #[Valid_tms, Node, Node, 1]

    def load_optimal_routing_mlu(self):
        self.optimal_mlu_file = self.traffic_file+'_optimal_mlu_per_tm'
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

    def get_optimal_routing_mlu(self):
        if self.load_optimal_routing_mlu() == False:
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
        assert next_hops_cnt > 0, (dst, visited_nodes)

        for np in next_hops:
            if np not in visited_nodes:
                self.traverse_next_hop(solution, np, dst, visited_nodes[:]+[np], affected_entries)
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

    def eliminate_loops(self, tm_idx, solution, actions):
        tm = self.traffic_matrices[tm_idx]
        affected_entries = []
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d or tm[s][d] == 0:
                    continue
                     
                self.traverse_next_hop(solution, s, d, [s], affected_entries)

        selected_node_dests = []
        for a in actions:
            selected_node_dests.append(self.pair_idx_to_sd[a])

        affected_ecmp_entries = []
        for e in affected_entries:
            if e not in selected_node_dests:
                affected_ecmp_entries.append(e)

        #print('Affected ecmp entries:', len(affected_ecmp_entries), affected_ecmp_entries)

        return len(affected_ecmp_entries)

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

        m = Model('routing')
        m.Params.OutputFlag = 0

        next_hop_loads = m.addVars(next_hops, lb=0, name='next_hop_loads')
        link_load = m.addVars(links, name='link_load')
        r = m.addVar(name='congestion_ratio')

        m.addConstrs(
            (next_hop_loads.sum('*', i, d) - next_hop_loads.sum(i, '*', d) == -tm[i][d] for i, d in pairs_sd),
            "node_dest_constr1")
        m.addConstrs(
            (next_hop_loads.sum('*', d, d) == sum(tm[ps][pd] for ps, pd in pairs_sd if pd == d) for d in nodes),
            "node_dest_constr2")

        m.addConstrs(
            (link_load[e] == next_hop_loads.sum(self.link_idx_to_sd[e][0], self.link_idx_to_sd[e][1], '*') for e in
                links), "link_load_constr")

        m.addConstrs((link_load[e] <= self.link_capacities[e] * r for e in links), "congestion_ratio_constr")

        m.setObjective(r + OBJ_EPSILON * link_load.sum('*', '*', '*'), GRB.MINIMIZE)

        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        # print('Obj:', m.objVal)
        obj_r = r.X
        solution = m.getAttr('x', next_hop_loads)

        return obj_r, solution

    def optimal_routing_mlu(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        m = Model('routing')
        m.Params.OutputFlag = 0

        ratio = m.addVars(self.lp_pairs, self.lp_links, name='ratio')

        link_load = m.addVars(self.lp_links, name='link_load')

        r = m.addVar(name='congestion_ratio')

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

        m.addConstrs((link_load[e] <= self.lp_link_capacity[e]*r for e in self.lp_links), "congestion_ratio_constr")

        #m.setObjective(r, GRB.MINIMIZE)
        m.setObjective(r + OBJ_EPSILON*link_load.sum('*', '*'), GRB.MINIMIZE)

        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        #print('Obj:', m.objVal)
        obj_r = r.X
        solution = m.getAttr('x', ratio)

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

        links_dest = [(n, i, d) for n,d in nodes_dest for i in self.neighbors[n]]
        links_dest += [(n, i, d) for n,d in ecmp_nodes_dest for i in self.ecmp_next_hops[n,d]]

        m = Model('routing')
        m.Params.OutputFlag = 0
        
        link_loads_dest = m.addVars(links_dest, name='link_loads_dest')
        
        r = m.addVar(name='congestion_ratio')

        m.addConstrs((link_loads_dest[ld] >= 0 for ld in links_dest), "link_load_dest_constr")

        m.addConstrs((link_loads_dest.sum('*', n, d) - link_loads_dest.sum(n, '*', d) == -tm[n][d] for n,d in nodes_dest), "node_dest_constr1")
        
        m.addConstrs((link_loads_dest.sum('*', d, d) == sum(tm[s][d] for s in self.lp_nodes if s!=d) for d in self.lp_nodes), "node_dest_constr2")
        
        m.addConstrs(((link_loads_dest.sum('*', n, d) + tm[n][d])/len(self.ecmp_next_hops[n,d]) == link_loads_dest[n, i, d] for n,d in ecmp_nodes_dest for i in self.ecmp_next_hops[n,d]), "node_dest_constr3")
        
        m.addConstrs((link_loads_dest.sum(e[0], e[1], '*') <= self.lp_link_capacity[e]*r for e in self.lp_links), "congestion_ratio_constr")
        
        m.setObjective(r + OBJ_EPSILON*link_loads_dest.sum('*', '*', '*'), GRB.MINIMIZE)
        
        m.optimize()

        assert m.status == GRB.Status.OPTIMAL
        #print('Obj:', m.objVal)
        solution = m.getAttr('x', link_loads_dest)

        return r.X, solution

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

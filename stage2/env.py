from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import numpy as np
import networkx as nx
#from gurobipy import *
#import matplotlib.pyplot as plt

class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file
        self.shortest_paths_file = self.topology_file +'_shortest_paths'
        self.preconfig_paths_file = self.topology_file +'_preconfig_paths'
        self.max_paths_per_pair = config.max_paths_per_pair
        self.DG = nx.DiGraph()

        self.load_topology()
        self.get_max_hop_cnt()
        self.calculate_paths()
        self.get_node_info()
        #self.calculate_ospf_paths()

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':')+2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10)+2:])
        f.readline()
        self.link_sd_to_idx = {}
        self.link_idx_to_sd = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            self.link_sd_to_idx[(int(s),int(d))] = int(i)
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s),int(d),int(w))])
        
        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        #print('nodes: %d, links: %d\n'%(self.num_nodes, self.num_links))
    
    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                self.pair_idx_to_sd.append((s,d))
                self.pair_sd_to_idx[(s,d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx+3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s,d))
                        self.pair_sd_to_idx[(s,d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s)+'->'+str(d)+': '+str(self.shortest_paths[-1])
                        f.writelines(line+'\n')
        
        assert self.num_pairs == self.num_nodes*(self.num_nodes-1)
        f.close()

        # Preconfig paths
        self.preconfig_paths = []
        max_paths_per_pair = 0
        if os.path.exists(self.preconfig_paths_file):
            print('[*] Loading preconfig paths...', self.preconfig_paths_file)
            f = open(self.preconfig_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                self.pair_idx_to_sd.append((s,d))
                self.pair_sd_to_idx[(s,d)] = self.num_pairs
                self.num_pairs += 1
                self.preconfig_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.preconfig_paths[-1].append(node_path)
                    paths = paths[idx+3:]

                if max_paths_per_pair < len(self.preconfig_paths[-1]):
                    max_paths_per_pair = len(self.preconfig_paths[-1])
        else:
            print('[!] Calculating preconfig paths...')
            f = open(self.preconfig_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s,d))
                        self.pair_sd_to_idx[(s,d)] = self.num_pairs
                        self.num_pairs += 1
                        self.preconfig_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s)+'->'+str(d)+': '+str(self.preconfig_paths[-1])
                        f.writelines(line+'\n')
                        
                        if max_paths_per_pair < len(self.preconfig_paths[-1]):
                            max_paths_per_pair = len(self.preconfig_paths[-1])

        assert self.num_pairs == self.num_nodes*(self.num_nodes-1)
        f.close()

        if self.max_paths_per_pair == -1:
            self.max_paths_per_pair = max_paths_per_pair
        else:
            self.preconfig_paths = [_paths[:self.max_paths_per_pair] for _paths in self.preconfig_paths]

        print('pairs: %d, nodes: %d, links: %d, max_paths_per_pair: %d, max_hop_cnt: %d\n'\
                %(self.num_pairs, self.num_nodes, self.num_links, self.max_paths_per_pair, self.max_hop_cnt))

    def get_max_hop_cnt(self):
        self.max_hop_cnt = 0
        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s != d:
                    shortest_path = nx.shortest_path(self.DG, s, d)
                    hop_cnt = len(shortest_path)-1
                    if self.max_hop_cnt < hop_cnt:
                        self.max_hop_cnt = hop_cnt
                        #print(s, d, shortest_path)

    def get_node_info(self):
        self.neighbors = [[n for n in self.DG.neighbors(node)] for node in range(self.num_nodes)]
        
        self.node_capacity_bound = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            for n in self.neighbors[i]:
                self.node_capacity_bound[i][0] += self.link_capacities[self.link_sd_to_idx[i,n]]    #egress 
                self.node_capacity_bound[i][1] += self.link_capacities[self.link_sd_to_idx[n,i]]    #ingress
            #print(self.node_capacity_bound[i])

class Traffic(object):
    def __init__(self, config, num_nodes, node_capacity_bound, data_dir='./data/', is_training=False):
        if is_training:
            self.traffic_file = data_dir + config.topology_file + config.traffic_file
        else:
            self.traffic_file = data_dir + config.topology_file + config.test_traffic_file
        self.num_nodes = num_nodes
        self.node_capacity_bound = node_capacity_bound
        self.avg_matrices_num = config.avg_matrices_num
        self.load_traffic(config)

    def load_traffic(self, config):
        if os.path.exists(self.traffic_file):
            print('[*] Loading traffic matrices...', self.traffic_file)
            f = open(self.traffic_file, 'r')
            traffic_matrices = []
            for line in f:
                volumes = line.strip().split(' ')
                total_volume_cnt = len(volumes)
                assert total_volume_cnt == self.num_nodes*self.num_nodes
                matrix = np.zeros((self.num_nodes, self.num_nodes))
                for v in range(total_volume_cnt):
                    i = int(v/self.num_nodes)
                    j = v%self.num_nodes
                    if i != j:
                        matrix[i][j] = float(volumes[v])
                #print(matrix + '\n')
                traffic_matrices.append(matrix)

            f.close()
            self.traffic_matrices = np.array(traffic_matrices)
        else:
            self.generate_traffic(config)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('Traffic matrices dims: [%d, %d, %d]\n'%(tms_shape[0], tms_shape[1], tms_shape[2]))

        if self.avg_matrices_num > 1:
            #manipulate traffic matrices
            avg_tm_cnt = int(self.tm_cnt/self.avg_matrices_num)
            avg_traffic_matrices = np.zeros((avg_tm_cnt, self.num_nodes, self.num_nodes))
            for i in range(avg_tm_cnt):
                avg_matrix = np.zeros((self.num_nodes, self.num_nodes))
                for j in range(self.avg_matrices_num):
                    avg_matrix += self.traffic_matrices[self.avg_matrices_num*i+j]
                avg_traffic_matrices[i] = avg_matrix/self.avg_matrices_num
        
            self.traffic_matrices = avg_traffic_matrices
            self.tm_cnt = avg_tm_cnt
            tms_shape = self.traffic_matrices.shape
            print('Traffic matrices dims: [%d, %d, %d]\n'%(tms_shape[0], tms_shape[1], tms_shape[2]))
   
    #Generate traffic based on hose model
    def generate_traffic(self, config, seed=1000):
        assert config.random_tms > 0
        random_state = np.random.RandomState(seed=seed)
        self.traffic_matrices = np.zeros((config.random_tms, self.num_nodes, self.num_nodes))
        print('[*] Generating random traffic matrices...')
        for t in tqdm(range(config.random_tms), ncols=70):
            prob_row = random_state.dirichlet(np.ones(self.num_nodes-1), size=self.num_nodes)
            #prob_column = random_state.dirichlet(np.ones(self.num_nodes-1), size=self.num_nodes)
            for i in range(self.num_nodes):
                n = 0
                for j in range(self.num_nodes):
                    if i != j:
                        self.traffic_matrices[t][i][j] = prob_row[i][n]*self.node_capacity_bound[i][0]*random_state.uniform(0,1) / 10
                        n += 1
            for c in range(self.num_nodes):
                column_sum = sum(self.traffic_matrices[t,:,c])
                if column_sum > self.node_capacity_bound[c][1]:
                    print('column_sum', column_sum, self.node_capacity_bound[c][1])
                    self.traffic_matrices[t,:,c] / column_sum * self.node_capacity_bound[c][1]

        print('[*] Saving random traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'w+')
        for t in range(config.random_tms):
            line = ''
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    line += str(self.traffic_matrices[t][i][j]) + ' '
            f.writelines(line[:-1]+'\n')
        f.close()

class Environment(object):
    def __init__(self, config, is_training=False):
        self.data_dir = './data/'
        self.topology = Topology(config, self.data_dir)
        self.traffic = Traffic(config, self.topology.num_nodes, self.topology.node_capacity_bound, self.data_dir, is_training=is_training)
        self.traffic_matrices = self.traffic.traffic_matrices*100*8/300/1000    #kbps
        
        self.tm_cnt = self.traffic.tm_cnt
        self.traffic_file = self.traffic.traffic_file
        self.num_pairs = self.topology.num_pairs
        self.max_paths_per_pair = self.topology.max_paths_per_pair
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)
        self.preconfig_paths_node = self.topology.preconfig_paths
        self.preconfig_paths_link = self.convert_to_edge_path(self.preconfig_paths_node)
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.neighbors = self.topology.neighbors
        self.incoming_and_outgoing_links_per_node()

    def incoming_and_outgoing_links_per_node(self):
        self.incoming_links = [[] for _ in range(self.num_nodes)]
        self.outgoing_links = [[] for _ in range(self.num_nodes)]
        for s,d in self.link_sd_to_idx:
            self.incoming_links[d].append((s,d))
            self.outgoing_links[s].append((s,d))

        self.node_capacity_bound = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            for e in self.outgoing_links[i]:
                self.node_capacity_bound[i][0] += self.link_capacities[self.link_sd_to_idx[i,e[1]]]    #egress
            for e in self.incoming_links[i]:
                self.node_capacity_bound[i][1] += self.link_capacities[self.link_sd_to_idx[e[0],i]]    #ingress

    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len-1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n+1])]
                    assert e>=0 and e<self.num_links
                    edge_paths[i][j].append(e)
                #print(i, j, edge_paths[i][j])

        return edge_paths

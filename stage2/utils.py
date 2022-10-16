from __future__ import print_function

import numpy as np
import multiprocessing as mp

def agent(i, func, tm_subset, mp_queue, ret_idx, ret_name):
    results = []
    tm_cnt = len(tm_subset)
    for idx in range(tm_cnt):
        tm_idx = tm_subset[idx]
        ret = func[0](tm_idx)
        if len(func) > 1:
            ret = func[1](ret[ret_idx])
            results.append(ret)
        else:
            results.append(ret[ret_idx])

        if idx == tm_cnt-1:
            print('Agent %d calculated %s %d(%d)'%(i, ret_name, idx, tm_cnt))
    
    mp_queue.put(results)

def multi_processing(func, tm_indexes, ret_idx, ret_name, num_agents=20):   ##changed agent
    total_results = []
    mp_queues = []
    if num_agents <= 0:
        num_agents = mp.cpu_count() - 1
    print('agent num: %d\n'%(num_agents))
    for _ in range(num_agents):
        mp_queues.append(mp.Queue(1))

    tm_subsets = np.array_split(tm_indexes, num_agents)

    agents = []
    for i in range(num_agents):
        agents.append(mp.Process(target=agent, args=(i, func, tm_subsets[i], mp_queues[i], ret_idx, ret_name)))
        agents[i].start()

    for i in range(num_agents):
        tm_cnt = len(tm_subsets[i])
        results = mp_queues[i].get()
      
        assert len(results) == tm_cnt, (i, len(results))
        total_results += results

    return total_results

# FlexEntry: Mitigating Routing Update Overhead for Traffic Engineering by Combining Destination-based Routing with Reinforcement Learning

This is a Tensorflow implementation of FlexEntry as described in our paper:

Minghao Ye, Yang Hu, Junjie Zhang, Zehua Guo, H. Jonathan Chao. (2021). Mitigating Routing Update Overhead for Traffic Engineering by Combining Destination-based Routing with Reinforcement Learning. Manuscript submitted for publication.

# Prerequisites

- Please install prerequisites (test with Ubuntu 20.04, Python 3.8.5, Tensorflow v2.2.0, Gurobi 9.1.1, networkx 2.5, tqdm 4.51.0)

# Training

- For stage 1, to train a sub-model for a topology, unzip and put the topology file (e.g., Ebone) and the traffic matrix file (e.g., EboneTM) in `stage1/data/`, then specify the file name in config.py, i.e., topology_file = 'Ebone' and traffic_file = 'TM', and then run 
```
python3 stage1/train.py
``` 
- For stage 2, to train a stage 2 model for a topology, unzip and put the topology file (e.g., Ebone) and the traffic matrix file (e.g., EboneTM) in `stage1/data/`, then specify the file name in config.py, i.e., topology_file = 'Ebone' and traffic_file = 'TM', and then run 
```
python3 stage2/train.py
``` 
- In a traffic matrix file, each line belongs to a N*N traffic matrix, where N is the node number of a topology.
- Please refer to `stage1/config.py` and `stage2/config.py` for more details about configurations. 

# Testing

- To test the sub-model in stage 1 on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and then run 
```
python3 stage1/test.py
```

- To test the model in stage 2 on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and put the sub-models in `stage2/models`, then run 
```
python3 stage2/test.py
```

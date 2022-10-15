# FlexEntry: Mitigating Routing Update Overhead for Traffic Engineering by Combining Destination-based Routing with Reinforcement Learning

This is a Tensorflow implementation of FlexEntry as described in our paper:

Minghao Ye, Yang Hu, Junjie Zhang, Zehua Guo, H. Jonathan Chao, "[Mitigating Routing Update Overhead for Traffic Engineering by Combining Destination-based Routing with Reinforcement Learning] (https://doi.org/10.1109/JSAC.2022.3191337)," in IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2662-2677, Sept. 2022, doi: 10.1109/JSAC.2022.3191337.

# Prerequisites

- Please install prerequisites (test with Ubuntu 20.04, Python 3.8.5, Tensorflow v2.2.0, Gurobi 9.1.1, networkx 2.5, tqdm 4.51.0)

# Training

- For stage 1, to train a sub-model for a topology, unzip and put the topology file (e.g., Ebone) and the traffic matrix file (e.g., EboneTM) in `stage1/data/`, then specify the file name in config.py, i.e., topology_file = 'Ebone' and traffic_file = 'TM', and then run 
```
python3 stage1/train.py
``` 
- For stage 2, to train a stage 2 model for a topology, unzip and put the topology file (e.g., Ebone) and the traffic matrix file (e.g., EboneTM) in `stage2/data/`, then specify the file name in config.py, i.e., topology_file = 'Ebone' and traffic_file = 'TM', and then run 
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

# Reference

Please cite our paper if you find our paper/code is useful for your work.

@ARTICLE{ye2022flexentry,
  author={Ye, Minghao and Hu, Yang and Zhang, Junjie and Guo, Zehua and Chao, H. Jonathan},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Mitigating Routing Update Overhead for Traffic Engineering by Combining Destination-Based Routing With Reinforcement Learning}, 
  year={2022},
  volume={40},
  number={9},
  pages={2662-2677},
  doi={10.1109/JSAC.2022.3191337}}
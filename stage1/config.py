class NetworkConfig(object):
  scale = 100

  max_step = 1000 * scale
  
  initial_learning_rate = 0.00001
  learning_rate_minimum = 0.000001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 5 * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.01

  save_step = 10 * scale
  max_to_keep = 1000

  #Transformer
  embedding_dim = 64
  LSTM_embedding = False
  num_layers = 6
  num_attention_heads = 8
  intermediate_dim = 256
  transformer_normalization = 'LayerNormalization'
  position_embedding = True

  #Conv
  Conv2D_out = 128
  Dense_out = 128
  batch_norm = False
  
  optimizer = 'RMSprop'
  #optimizer = 'Adam'
    
  logit_clipping = 10           #10 or 0, = 0 means logit clipping is disabled

  l2_regularizer = 0

class Config(NetworkConfig):
  version = 'TE_3.2'

  project_name = 'FlexEntry'

  # topology_file = 'Abilene'
  topology_file = 'Ebone'
  # topology_file = 'Sprintlink'
  # topology_file = 'Tiscali'
  # topology_file = 'nobel'
  # topology_file = 'Germany50'

  # Critical entries number
  # Change according to topology
  max_moves = 51

  method = 'actor_critic'
  
  model_type = 'Conv'
  
  encoder = 'Graph'

  reward_type = 'mlu'

  dis_thresh = 0.1

  avg_matrices_num = 1

  traffic_file = 'TM1'
  test_traffic_file = 'TM2'

  tm_history = 1
  predict_interval = 0

  num_agents = 20



  normalized_reward = True

  softmax_temperature = False

  # For pure policy
  baseline = 'avg'          #avg, best

  num_iter = 20
  max_paths_per_pair = 4   # = -1 means all paths are selected

  LP_solver = 'Gurobi'

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config

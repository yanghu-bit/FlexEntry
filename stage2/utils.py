from __future__ import print_function

import math
import six
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    raise ValueError(
        "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
        "equal to the expected tensor rank `%s`" %
        (name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal

def agent(i, func, tm_subset, mp_queue, ret_idx, ret_name):
    results = []
    tm_cnt = len(tm_subset)
    freq = tm_cnt//20
    for idx in range(tm_cnt):
        tm_idx = tm_subset[idx]
        ret = func[0](tm_idx)
        if len(func) > 1:
            ret = func[1](ret[ret_idx])
            results.append(ret)
        else:
            results.append(ret[ret_idx])

        #if idx % freq == 0 or idx == tm_cnt-1:
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum()


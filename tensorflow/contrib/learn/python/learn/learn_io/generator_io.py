# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods to allow generator of dict with numpy arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from types import FunctionType, GeneratorType
from collections import OrderedDict

from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_functions


def generator_input_fn(x,
                       target_key=None,
                       batch_size=128,
                       num_epochs=1,
                       shuffle=True,
                       queue_capacity=1000,
                       num_threads=1):
  """Returns input function that would dicts of numpy arrays yielded from a generator.
  
  It is assumed that every dict yielded from the dictionary represents
  a single sample. The generator should consume a single epoch of the data.

  This returns a function outputting `features` and `target` based on the dict
  of numpy arrays. The dict `features` has the same keys as an element yielded
  from x.

  Example:
    ```python
    def generator():
      for index in range(10):
        yield {height: np.random.randint(32,36), 'age': np.random.randint(18, 80),
              "label": np.ones(1)}

    with tf.Session() as session:
      input_fn = generator_io.generator_input_fn(
          generator, target_key="label", batch_size=2, shuffle=False,
          num_epochs=1)
    ```

  Args:
    x: Generator Function, returns a `Generator` that will yield the data
      in `dict` of numpy arrays
    target_key: String or List of Strings, the key or list of keys of
      the numpy arrays in x dictionaries to use as target.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
      time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing.

  Returns:
    Function, that returns a feature `dict` with `Tensors` and
     an optional label `dict` with `Tensors`

  Raises:
    TypeError: `x` is not `FunctionType`.
    TypeError: `x()` is not `GeneratorType`.
    TypeError: `next(x())` is not `dict`.
    TypeError: `target_key` is not `str` or `target_key` is not `list` of `str`.
    KeyError:  `target_key` not a key or `target_key[index]` not in next(`x()`).
  """
  
  def _generator_input_fn():
    """generator input function."""
    if not isinstance(x, FunctionType):
      raise TypeError('x must be generator function ; got {}'.format(type(x).__name__))
    generator = x()
    if not isinstance(generator, GeneratorType):
      raise TypeError('x() must be generator ; got {}'.format(type(generator).__name__))
    data = next(generator)
    if not isinstance(data, dict):
      raise TypeError('x() must yield dict ; got {}'.format(type(data).__name__))
    
    input_keys = next(x()).keys()
    if target_key is not None:
      if isinstance(target_key, str):
        target_key = [target_key]
      if isinstance(target_key, list):
        for item in target_key:
          if not isinstance(item, str):
            raise TypeError(
              'target_key must be str or list of str ; got {}'.format(type(item).__name__))
          if item not in input_keys:
            raise KeyError('target_key or target_key[i] not in yielded dict ; got {}'.format(item))

    queue = feeding_functions.enqueue_data(
      x,
      queue_capacity,
      shuffle=shuffle,
      num_threads=num_threads,
      enqueue_size=batch_size,
      num_epochs=num_epochs)
    
    features = (queue.dequeue_many(batch_size) if num_epochs is None
                else queue.dequeue_up_to(batch_size))
    
    features = dict(zip(input_keys, features))
    if target_key is not None:
      target = {key: features.pop(key) for key in target_key}
      return features, target
    return features
  return _generator_input_fn

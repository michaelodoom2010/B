# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.layers.pooling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint

from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class PoolingTest(test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'data_format'):
      pooling_layers.max_pooling2d(images, 3, strides=2, data_format='invalid')

  def testInvalidStrides(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'strides'):
      pooling_layers.max_pooling2d(images, 3, strides=(1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'strides'):
      pooling_layers.max_pooling2d(images, 3, strides=None)

  def testInvalidPoolSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'pool_size'):
      pooling_layers.max_pooling2d(images, (1, 2, 3), strides=2)

    with self.assertRaisesRegexp(ValueError, 'pool_size'):
      pooling_layers.max_pooling2d(images, None, strides=2)

  def testCreateMaxPooling2D(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = pooling_layers.MaxPooling2D([2, 2], strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4, 4])

  def testCreateAveragePooling2D(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = pooling_layers.AveragePooling2D([2, 2], strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4, 4])

  def testCreateMaxPooling2DChannelsFirst(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, 2, height, width))
    layer = pooling_layers.MaxPooling2D([2, 2],
                                        strides=1,
                                        data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 6, 8])

  def testCreateAveragePooling2DChannelsFirst(self):
    height, width = 5, 6
    images = random_ops.random_uniform((3, 4, height, width))
    layer = pooling_layers.AveragePooling2D((2, 2),
                                            strides=(1, 1),
                                            padding='valid',
                                            data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [3, 4, 4, 5])

  def testCreateAveragePooling2DChannelsFirstWithNoneBatch(self):
    height, width = 5, 6
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, 4, height, width))
    layer = pooling_layers.AveragePooling2D((2, 2),
                                            strides=(1, 1),
                                            padding='valid',
                                            data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [None, 4, 4, 5])

  def testCreateMaxPooling1D(self):
    width = 7
    images = random_ops.random_uniform((5, width, 4))
    layer = pooling_layers.MaxPooling1D(2, strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4])

  def testSpatialPyramidPoolingDefaultDimensionForBins(self):
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling()
    output = layer.apply(images)
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingCustomDimensionForBins(self):
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingBatchSizeGiven(self):
    batch_size, height, width, channel = 4, 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [batch_size, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingAssertOutDimensionFixedForAnyInput(self):
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output_arrays = []
    check_for_images = 10
    batch_size, channel = 2, 3
    for _ in range(check_for_images):
      height, width = randint(0, 9), randint(0, 9)
      images = array_ops.placeholder(dtype='float32',
                                     shape=(batch_size, height, width, channel))
      output = layer.apply(images)
      output_arrays.append(output.get_shape().as_list())
    self.assertListEqual(output_arrays,
                         [[batch_size, channel * expected_output_size_for_each_channel]] * check_for_images)

  def testCreateAveragePooling1D(self):
    width = 7
    images = random_ops.random_uniform((5, width, 4))
    layer = pooling_layers.AveragePooling1D(2, strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4])

  def testCreateMaxPooling1DChannelsFirst(self):
    width = 7
    images = random_ops.random_uniform((5, width, 4))
    layer = pooling_layers.MaxPooling1D(
        2, strides=2, data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4])

  def testCreateAveragePooling1DChannelsFirst(self):
    width = 7
    images = random_ops.random_uniform((5, width, 4))
    layer = pooling_layers.AveragePooling1D(
        2, strides=2, data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4])

  def testCreateMaxPooling3D(self):
    depth, height, width = 6, 7, 9
    images = random_ops.random_uniform((5, depth, height, width, 4))
    layer = pooling_layers.MaxPooling3D([2, 2, 2], strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 4, 4])

  def testCreateAveragePooling3D(self):
    depth, height, width = 6, 7, 9
    images = random_ops.random_uniform((5, depth, height, width, 4))
    layer = pooling_layers.AveragePooling3D([2, 2, 2], strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 4, 4])

  def testMaxPooling3DChannelsFirst(self):
    depth, height, width = 6, 7, 9
    images = random_ops.random_uniform((5, 2, depth, height, width))
    layer = pooling_layers.MaxPooling3D(
        [2, 2, 2], strides=2, data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 3, 3, 4])

  def testAveragePooling3DChannelsFirst(self):
    depth, height, width = 6, 7, 9
    images = random_ops.random_uniform((5, 2, depth, height, width))
    layer = pooling_layers.AveragePooling3D(
        [2, 2, 2], strides=2, data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 3, 3, 4])

  def testCreateMaxPooling2DIntegerPoolSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = pooling_layers.MaxPooling2D(2, strides=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 4, 4])

  def testMaxPooling2DPaddingSame(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4), seed=1)
    layer = pooling_layers.MaxPooling2D(
        images.get_shape()[1:3], strides=2, padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 4, 5, 4])

  def testCreatePooling2DWithStrides(self):
    height, width = 6, 8
    # Test strides tuple
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    layer = pooling_layers.MaxPooling2D([2, 2], strides=(2, 2), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 3])

    # Test strides integer
    layer = pooling_layers.MaxPooling2D([2, 2], strides=2, padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 3])

    # Test unequal strides
    layer = pooling_layers.MaxPooling2D([2, 2], strides=(2, 1), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width, 3])


if __name__ == '__main__':
  test.main()

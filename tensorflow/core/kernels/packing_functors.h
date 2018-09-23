/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename Device, typename T>
class PackedSequenceAlignmentOp;

template <typename Device, typename T>
class SequenceGatherScatterIndicesOp;

template <typename Device, typename T, typename Index>
class PackSequenceOp;

template <typename Device, typename T, typename Index>
class UnpackSequenceOp;

namespace functor {

template <typename Device, typename T>
struct PackedSequenceAlignmentFunctor {
  Status operator()(
	const Device& d, 
   typename TTypes<T>::ConstFlat Tsequence_lengths,
   typename TTypes<T>::Flat Talignments,
   typename TTypes<T>::Flat Tbatch_sizes);
};

template <typename Device, typename T>
struct SequenceGatherScatterIndicesFunctor {
  Status operator()(
	const Device& d, 
   typename TTypes<T>::ConstFlat Tsequence_lengths,
   typename TTypes<T>::ConstFlat Tbatch_order,
   typename TTypes<T>::Flat Tgather_scatter_indices,
   bool time_major);
};

template <typename Device, typename T, typename Index>
struct PackSequenceFunctor {
  Status operator()(
	const Device& d, 
   typename TTypes<T,3>::ConstTensor Tsequence,
   typename TTypes<Index>::ConstFlat Talignments,
   typename TTypes<Index>::ConstFlat Tbatch_sizes,
   typename TTypes<T,2>::Tensor Tpacked
   );
};

template <typename Device, typename T, typename Index>
struct UnpackSequenceFunctor {
  Status operator()(
	const Device& d, 
   typename TTypes<T,2>::ConstTensor Tpacked,
   typename TTypes<Index>::ConstFlat Talignments,
   typename TTypes<Index>::ConstFlat Tbatch_sizes,
   typename TTypes<T,3>::Tensor Tsequence
   );
};


}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_

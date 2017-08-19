/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "external/cub_archive/cub/device/device_reduce.cuh"
#include "external/cub_archive/cub/device/device_segmented_reduce.cuh"
#include "external/cub_archive/cub/iterator/counting_input_iterator.cuh"
#include "external/cub_archive/cub/iterator/transform_input_iterator.cuh"
#include "external/cub_archive/cub/warp/warp_reduce.cuh"
#include "cuda/include/cuComplex.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/permutation_input_iterator.h"
#include "tensorflow/core/util/transform_output_iterator.h"

#include <sstream>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct Prod {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded multiply op for std::complex
template <>
struct Prod<std::complex<float>> {
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& a, const std::complex<float>& b) const {
    auto result = cuCmulf(make_cuComplex(a.real(), a.imag()),
                          make_cuComplex(b.real(), b.imag()));
    return std::complex<float>(result.x, result.y);
  }
};

template <>
struct Prod<std::complex<double>> {
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& a, const std::complex<double>& b) const {
    auto result = cuCmul(make_cuDoubleComplex(a.real(), a.imag()),
                         make_cuDoubleComplex(b.real(), b.imag()));
    return std::complex<double>(result.x, result.y);
  }
};

template <typename T, typename outT = T>
struct DividesBy {
  T divisor;

  __host__ __device__ explicit DividesBy(T divisor) : divisor(divisor) {}

  __host__ __device__ outT operator()(const T& x) const { return x / divisor; }
};

// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded ops for std::complex
template <>
struct DividesBy<std::complex<float>> {
  cuFloatComplex divisor;

  __host__ __device__ explicit DividesBy(std::complex<float> divisor)
      : divisor(make_cuComplex(divisor.real(), divisor.imag())) {}

  // implements
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& x) const {
    auto result = cuCdivf(make_cuComplex(x.real(), x.imag()), divisor);
    return std::complex<float>(result.x, result.y);
  }
};

template <>
struct DividesBy<std::complex<double>> {
  cuDoubleComplex divisor;

  __host__ __device__ explicit DividesBy(std::complex<double> divisor)
      : divisor(make_cuDoubleComplex(divisor.real(), divisor.imag())) {}

  // implements
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& x) const {
    auto result = cuCdiv(make_cuDoubleComplex(x.real(), x.imag()), divisor);
    return std::complex<double>(result.x, result.y);
  }
};

template <>
struct DividesBy<float, Eigen::half> {
  float divisor;

  __host__ __device__ explicit DividesBy(float divisor) : divisor(divisor) {}

  __host__ __device__ Eigen::half operator()(const float& x) const {
    return Eigen::half(x / divisor);
  }
};

struct HalfToFloat {
  __host__ __device__ float operator()(const Eigen::half& x) const {
    return Eigen::half_impl::half_to_float(x);
  }
};

struct FloatToHalf {
  __host__ __device__ Eigen::half operator()(const float& x) const {
    return Eigen::half_impl::float_to_half_rtne(x);
  }
};

struct And {
  __host__ __device__ bool operator()(const bool& a, const bool& b) const {
    return a && b;
  }
};

struct Or {
  __host__ __device__ bool operator()(const bool& a, const bool& b) const {
    return a || b;
  }
};

// each block does a grid strided loop and reduces its values locally
// the case of one block is used for low latency small reductions to scalars
template <typename T, typename outT, int num_threads, typename Op>
__global__ void BlockReduceKernel(T in, outT out, int num_elems, Op op) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int gid = bid * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  typedef typename std::iterator_traits<T>::value_type value_type;

  value_type sum;
  if (gid < num_elems) {
    sum = in[gid];
    for (int pos = gid + stride; pos < num_elems; pos += stride) {
      sum = op(sum, in[pos]);
    }
  } else
    sum = value_type();  // stop compiler from complaining

  typedef cub::BlockReduce<value_type, num_threads> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  __syncthreads();

  sum = BlockReduce(temp_storage)
            .template Reduce(sum, op, min(num_elems, num_threads));

  if (tid == 0) out[bid] = sum;
}

// maps a warp to each row
template <typename T, typename outT, typename Op>
__global__ void RowReduceKernel(T in, outT out, int num_rows, int num_cols,
                                Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;

  if (num_cols == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < num_rows) out[gid] = in[gid];
    return;
  }

  value_type sum;
  int col = lane;
  if (row < num_rows && col < num_cols) {
    sum = in[row * num_cols + col];
    col += 32;
    for (; col < num_cols; col += 32) {
      sum = op(sum, in[row * num_cols + col]);
    }
  } else {
    sum = value_type();  // stop compiler from complaining
  }

  typedef cub::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  __syncthreads();

  sum = WarpReduce(temp_storage).template Reduce(sum, op, min(num_cols, 32));

  if (row < num_rows && lane == 0) out[row] = sum;
}

// Works only if there are <= 16 columns
// each warps sums over multiple rows at once
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceMax16ColumnsKernel(T in, outT out, int num_rows,
                                               int num_cols, Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int rows_per_warp = 32 / num_cols;

  int lane = threadIdx.x % 32;
  int lane_row = lane / num_cols;

  const int start_row_warp =
      rows_per_warp * (blockIdx.y * blockDim.y + threadIdx.y);
  const int start_row_lane = start_row_warp + lane_row;
  int row = start_row_lane;
  int col = lane % num_cols;

  value_type sum;
  if (row * num_cols + col < num_rows * num_cols)
    sum = in[row * num_cols + col];
  else
    sum = value_type();  // needed to shut up compiler

  __shared__ value_type partial_sums[32][33];

  __syncthreads();

  row += rows_per_warp * gridDim.y * blockDim.y;
  for (; row < num_rows; row += rows_per_warp * gridDim.y * blockDim.y) {
    int global_pos = row * num_cols + col;
    if (global_pos < (num_rows * num_cols))
      sum = op(sum, in[row * num_cols + col]);
  }

  const int rows_in_this_warp = min(rows_per_warp, num_rows - start_row_warp);
  // not the most efficient way to do this sum
  for (int i = 1; i < rows_in_this_warp; ++i) {
    value_type tmp =
        cub::ShuffleIndex(sum, threadIdx.x + i * num_cols, 32, 0xffffffff);
    if (lane < num_cols) sum = op(sum, tmp);
  }

  if (lane < num_cols) partial_sums[lane][threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < num_cols) {
    value_type s = partial_sums[threadIdx.x][0];

    if (blockDim.y > 1) {
      for (int row = 1; row < blockDim.y; ++row) {
        s = op(s, partial_sums[threadIdx.x][row]);
      }
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// Maps each block to a column range 32 wide
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceKernel(T in, outT out, int num_rows, int num_cols,
                                   Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * 32 + threadIdx.x;

  value_type sum;
  if (row * num_cols + col < num_rows * num_cols)
    sum = in[row * num_cols + col];
  else
    sum = value_type();  // will never be used, needed to shut up compiler

  __shared__ value_type partial_sums[32][33];

  __syncthreads();

  row += gridDim.y * blockDim.y;

  if (col < num_cols) {
    for (; row < num_rows; row += gridDim.y * blockDim.y) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  partial_sums[threadIdx.x][threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < 32) {
    value_type s = partial_sums[threadIdx.x][0];

    for (int row = 1; row < blockDim.y; ++row) {
      s = op(s, partial_sums[threadIdx.x][row]);
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// does multiple warp size segmented reductions in parallel
// segments cannot cross warp boundaries (mainly used for reducing the segments
// that come from the Max16Columns column reduction kernel)
template <typename T, typename outT, typename Op>
__global__ void CleanupSegments(T partial_sums, outT out, int num_rows,
                                int num_cols, int segment_size, Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  value_type val;
  if (tid < segment_size * num_cols)
    val = partial_sums[tid];
  else
    val = value_type();  // 0s beyond last segment won't be used, so OK

  typedef cub::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  __syncthreads();

  bool head_flag = (threadIdx.x % segment_size) == 0;
  value_type sum =
      WarpReduce(temp_storage).HeadSegmentedReduce(val, head_flag, op);

  if (head_flag && tid < segment_size * num_cols) {
    out[tid / segment_size] = sum;
  }
}

// assigns one thread to a column
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceSimpleKernel(T in, outT out, int num_planes,
                                         int num_rows, int num_cols, Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  int plane = gid / num_cols;
  int col = gid % num_cols;

  if (plane >= num_planes) return;

  if (num_rows == 1) {
    out[plane * elems_per_plane + col] = in[plane * elems_per_plane + col];
    return;
  }

  value_type sum = op(in[plane * elems_per_plane + col],
                      in[plane * elems_per_plane + num_cols + col]);
  for (int row = 2; row < num_rows; ++row) {
    sum = op(sum, in[plane * elems_per_plane + row * num_cols + col]);
  }

  out[plane * num_cols + col] = sum;
}

struct RowOffset {
  __host__ __device__ explicit RowOffset(const int& cols) : cols_(cols) {}

  __host__ __device__ int operator()(const int& x) const { return cols_ * x; }

  int cols_;
};

struct GatherOp {
  __host__ __device__ GatherOp(const int& extent_x, const int& extent_y,
                               const int& extent_z, bool kOne)
      : extent_x_(extent_x),
        extent_y_(extent_y),
        extent_z_(extent_z),
        kOne_(kOne) {
    if (kOne_)
      group_size_ = extent_y_;
    else
      group_size_ = extent_x_ * extent_z_;
  }

  __host__ __device__ int operator()(const int& ind) const {
    const int group = kOne_ ? ind / group_size_ : ind % group_size_;
    const int offset = kOne_ ? ind % group_size_ : ind / group_size_;

    const int x = group / extent_z_;
    const int z = group % extent_z_;

    return x * extent_y_ * extent_z_ + z + offset * extent_z_;
  }

  int extent_x_;
  int extent_y_;
  int extent_z_;
  bool kOne_;
  int group_size_;
};

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchScalarReduction(OpKernelContext* ctx, OUT_T out, IN_T in,
                           int in_size, Op op, T init,
                           const cudaStream_t& cu_stream) {
  // handle situations where low latency is important better than CUB
  if (in_size <= 4096) {
    const int num_blocks = 1;
    const int num_threads = 256;
    BlockReduceKernel<IN_T, OUT_T, num_threads>
        <<<num_blocks, num_threads, 0, cu_stream>>>(in, out, in_size, op);
    return;
  } else if (in_size <= 1 << 19) {
    const int num_threads = 256;
    const int num_blocks = 32;  // it seems like tailoring this to the GPU
                                // would be more effective, but all attempts
                                // at making this a multiple of the number of
                                // multiprocessors have lead to lower perf
                                // in general
                                // TODO(eriche) investigate this more

    Tensor temp_storage;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64>(num_blocks * sizeof(T))}),
            &temp_storage));

    BlockReduceKernel<IN_T, T*, num_threads>
        <<<num_blocks, num_threads, 0, cu_stream>>>(
            in, (T*)temp_storage.flat<int8_t>().data(), in_size, op);

    CleanupSegments<<<1, num_blocks, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, 1, 1, num_blocks, op);
    return;
  }
  std::size_t temp_storage_bytes = 0;

  Tensor temp_storage;
  // written as a loop because it reduces clutter
  // first pass allocates memory, second launches kernel(s)
  for (int i = 0; i < 2; ++i) {
    auto success = cub::DeviceReduce::Reduce(
        i == 0 ? nullptr : temp_storage.flat<int8_t>().data(),
        temp_storage_bytes, in, out, in_size, op, init, cu_stream);

    OP_REQUIRES(
        ctx, success == 0,
        errors::Internal("CUB reduce error", cudaGetErrorString(success)));

    if (i == 0)
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &temp_storage));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchRowReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int num_rows,
                        int num_cols, Op op, T init,
                        const cudaStream_t& cu_stream) {
  if (num_cols < 1024) {
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    RowReduceKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
        in, out, num_rows, num_cols, op);
    return;
  }

  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(num_cols);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, RowOffset, cub::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  std::size_t temp_storage_bytes = 0;
  Tensor temp_storage;
  for (int i = 0; i < 2; ++i) {
    auto success = cub::DeviceSegmentedReduce::Reduce(
        i == 0 ? nullptr : temp_storage.flat<int8_t>().data(),
        temp_storage_bytes, in, out, num_rows, transform_iter,
        transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 cudaGetErrorString(success)));

    if (i == 0)
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &temp_storage));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction_LTE16Cols(OpKernelContext* ctx, OUT_T out, IN_T in,
                                     int extent_x, int extent_y, Op op, T init,
                                     const cudaStream_t& cu_stream) {
  int rows_per_warp = 32 / extent_y;
  dim3 block_dim(32, min(Eigen::divup(extent_x, rows_per_warp), 32), 1);
  dim3 grid_dim(1,
                Eigen::divup(static_cast<unsigned int>(extent_x),
                             rows_per_warp * block_dim.y),
                1);

  grid_dim.y = min((int)grid_dim.y, 32);

  if (grid_dim.y > 2 && grid_dim.y < 32) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    ColumnReduceMax16ColumnsKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, out, extent_x, extent_y, op);
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));
    ColumnReduceMax16ColumnsKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, (T*)temp_storage.flat<int8_t>().data(), extent_x, extent_y, op);

    dim3 new_grid_dim((grid_dim.y * extent_y + 31) / 32, 1, 1);
    dim3 num_threads(128, 1, 1);
    CleanupSegments<<<new_grid_dim, block_dim, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, extent_x, extent_y,
        grid_dim.y, op);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction_LTE4096Cols(OpKernelContext* ctx, OUT_T out, IN_T in,
                                       int extent_x, int extent_y, Op op,
                                       T init, const cudaStream_t& cu_stream) {
  dim3 block_dim(32, min(extent_x, 32), 1);
  dim3 grid_dim((extent_y + 31) / 32, 1, 1);

  if (grid_dim.x < 16) grid_dim.y = min((extent_x + 31) / 32, 32);

  if (grid_dim.y > 2 && grid_dim.y < 32) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    ColumnReduceKernel<<<grid_dim, block_dim, 0, cu_stream>>>(in, out, extent_x,
                                                              extent_y, op);
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));

    ColumnReduceKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, (T*)temp_storage.flat<int8_t>().data(), extent_x, extent_y, op);

    dim3 new_grid_dim((grid_dim.y * extent_y + 31) / 32, 1, 1);
    dim3 num_threads(128, 1, 1);
    CleanupSegments<<<new_grid_dim, block_dim, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, extent_x, extent_y,
        grid_dim.y, op);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction(OpKernelContext* ctx, OUT_T out, IN_T in,
                           int extent_x, int extent_y, Op op, T init,
                           const cudaStream_t& cu_stream) {
  if (extent_y <= 16) {
    LaunchColumnReduction_LTE16Cols(ctx, out, in, extent_x, extent_y, op, init,
                                    cu_stream);
  } else if (extent_y <= 4096) {
    LaunchColumnReduction_LTE4096Cols(ctx, out, in, extent_x, extent_y, op,
                                      init, cu_stream);
  } else {
    int threads_per_block = 128;
    int num_blocks = Eigen::divup(extent_y, threads_per_block);

    ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
        in, out, 1, extent_x, extent_y, op);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DYReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                        int extent_y, int extent_z, Op op, T init,
                        const cudaStream_t& cu_stream) {
  int threads_per_block = 128;
  int num_blocks =
      (extent_x * extent_z + threads_per_block - 1) / threads_per_block;

  // TODO (eriche): this won't be very good in the case of small x
  //                small z and large y.
  ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
      in, out, extent_x, extent_y, extent_z, op);
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DXZReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                         int extent_y, int extent_z, Op op, T init,
                         const cudaStream_t& cu_stream) {
  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(extent_x * extent_z);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, RowOffset, cub::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  GatherOp gather_op(extent_x, extent_y, extent_z, false);
  typedef cub::TransformInputIterator<int, GatherOp,
                                      cub::CountingInputIterator<int>>
      gatherIterType;
  gatherIterType gather_iter(counting_iter, gather_op);

  PermutationInputIterator<T, IN_T, gatherIterType> permute_iter(in,
                                                                 gather_iter);

  std::size_t temp_storage_bytes = 0;
  Tensor temp_storage;

  for (int i = 0; i < 2; ++i) {
    auto success = cub::DeviceSegmentedReduce::Reduce(
        i == 0 ? nullptr : temp_storage.flat<int8_t>().data(),
        temp_storage_bytes, permute_iter, out, extent_y, transform_iter,
        transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 cudaGetErrorString(success)));

    if (i == 0)
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &temp_storage));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T,
          typename ReductionAxes>
void ReduceImpl(OpKernelContext* ctx, OUT_T out, IN_T in, int in_rank,
                int in_dim0, int in_dim1, int in_dim2, int out_rank,
                const ReductionAxes& reduction_axes, Op op, T init) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);
  if (out_rank == 0) {
    const int in_size = in_dim0 * in_dim1 * in_dim2;
    LaunchScalarReduction(ctx, out, in, in_size, op, init, cu_stream);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 1) {  // row reduction
    LaunchRowReduction(ctx, out, in, in_dim0, in_dim1, op, init, cu_stream);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 0) {  // column reduction
    LaunchColumnReduction(ctx, out, in, in_dim0, in_dim1, op, init, cu_stream);
  } else if (in_rank == 3 && out_rank == 2 && reduction_axes[0] == 1) {
    Launch3DYReduction(ctx, out, in, in_dim0, in_dim1, in_dim2, op, init,
                       cu_stream);
  } else if (in_rank == 3 && out_rank == 1 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2) {
    Launch3DXZReduction(ctx, out, in, in_dim0, in_dim1, in_dim2, op, init,
                        cu_stream);
  } else {
    std::stringstream ss;
    ss << "Invalid reduction requested: in_rank, out_rank, axes " << in_rank
       << " " << out_rank;
    if (out_rank == 1) ss << " " << reduction_axes[0];
    if (out_rank == 2) ss << " " << reduction_axes[1];
    LOG(FATAL) << ss.str();
  }
}

}  // namespace functor
}  // namespace tensorflow

#endif

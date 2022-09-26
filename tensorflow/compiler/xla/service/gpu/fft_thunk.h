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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FFT_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FFT_THUNK_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace gpu {

struct FftPlan {
  // CuFFT thread-safety requires that separate host threads not share plans;
  // protect each plan with a mutex.
  absl::Mutex mu;
  std::unique_ptr<se::fft::Plan> plan ABSL_GUARDED_BY(mu);
  float scale_factor ABSL_GUARDED_BY(mu);
};

class FftPlanCache {
 public:
  // Returnes Fft plan cached for the given device ordinal or creates a new one.
  FftPlan* GetOrCreate(int device_ordinal) {
    absl::MutexLock lock(&mu_);
    std::unique_ptr<FftPlan>& plan = fft_plans_[device_ordinal];
    if (!plan) plan = std::make_unique<FftPlan>();
    return plan.get();
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<int, std::unique_ptr<FftPlan>> fft_plans_
      ABSL_GUARDED_BY(mu_);
};

// This class stores everything that StreamExecutor needs to launch an FFT.
// It is generated by IrEmitter.
//
// This is thread-compatible.
class FftThunk : public Thunk {
 public:
  // Constructs a thunk for launching an FFT on a stream.
  // Semantics of null hlo_instruction argument are as in Thunk.
  FftThunk(ThunkInfo thunk_info, FftType fft_type,
           absl::Span<const int64_t> fft_length,
           const BufferAllocation::Slice& input_buffer,
           const BufferAllocation::Slice& output_buffer,
           const Shape& input_shape, const Shape& output_shape);

  FftThunk(const FftThunk&) = delete;             // Cannot share fft_plan_
  FftThunk& operator=(const FftThunk&) = delete;  // Cannot share fft_plan_

  // Does the FFT for the thunk on "stream".
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const se::fft::Type fft_type_;
  const std::vector<int64_t> fft_length_;

  FftPlanCache fft_plan_cache_;

  const BufferAllocation::Slice input_buffer_;
  const BufferAllocation::Slice output_buffer_;

  const Shape input_shape_;
  const Shape output_shape_;
};

Status RunFft(se::DeviceMemoryBase input, const Shape& input_shape,
              se::DeviceMemoryBase output, const Shape& output_shape,
              se::fft::Type fft_type, absl::Span<const int64_t> fft_length,
              int device_ordinal, FftPlanCache* fft_plan_cache,
              se::Stream* stream, se::DeviceMemoryAllocator* memory_allocator);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FFT_THUNK_H_

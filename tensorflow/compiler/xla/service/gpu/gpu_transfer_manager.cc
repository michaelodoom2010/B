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

#include "tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/DataLayout.h"
#include "tensorflow/compiler/xla/literal_util.h"
// XXX figure out how to cope with both platforms
#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h"
#endif
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// TODO(b/30467474) Once GPU infeed implementation settles, consider
// folding back the cpu and gpu infeed implementations into a generic
// one if possible.
GpuTransferManager::GpuTransferManager(se::Platform::Id id)
    : GenericTransferManager(
          id,
// XXX figure out how to cope with both platforms
#if GOOGLE_CUDA
          /*pointer_size=*/llvm::DataLayout(gpu::NVPTXCompiler::kDataLayout)
#elif TENSORFLOW_USE_ROCM
          /*pointer_size=*/llvm::DataLayout(gpu::AMDGPUCompiler::kDataLayout)
#endif
              .getPointerSize(0 /* default address space */)) {}

Status GpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(shape);

  if (!ShapeUtil::IsTuple(shape)) {
    int64 size = GetByteSizeRequirement(shape);
    return TransferBufferToInfeed(executor, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()).c_str());
  }

  // For a tuple, we transfer each of its elements to the device and
  // enqueue the resulting destination device addresses with the
  // infeed manager.
  std::vector<gpu::InfeedBuffer*> buffers;
  buffers.reserve(ShapeUtil::TupleElementCount(shape));
  auto cleanup = tensorflow::gtl::MakeCleanup([buffers]() {
    for (gpu::InfeedBuffer* b : buffers) {
      b->Done();
    }
  });

  for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(shape, i);
    int64 tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
    TF_ASSIGN_OR_RETURN(
        gpu::InfeedBuffer * buffer,
        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                       literal.untyped_data({i})));
    buffers.push_back(buffer);
  }

  cleanup.release();
  return EnqueueBuffersToInfeed(executor, buffers);
}

Status GpuTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                                  int64 size,
                                                  const void* source) {
  TF_ASSIGN_OR_RETURN(gpu::InfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(executor, size, source));
  return EnqueueBuffersToInfeed(executor, {buffer});
}

Status GpuTransferManager::EnqueueBuffersToInfeed(
    se::StreamExecutor* executor, std::vector<gpu::InfeedBuffer*> buffers) {
  gpu::InfeedManager* infeed_manager = gpu::GetOrCreateInfeedManager();
  se::Stream* stream = infeed_manager->GetStream(executor);

  // TODO(b/30467474): Since this stream is shared across different
  // infeed requests, blocking on the stream might be
  // heavy-handed. Figure out if finer-grained acknowledgement is
  // possible.
  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) {
    for (gpu::InfeedBuffer* b : buffers) {
      b->Done();
    }
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         stream, block_status.error_message().c_str());
  }

  infeed_manager->EnqueueBuffers(buffers);

  VLOG(2) << "Infeed data transferred";

  return Status::OK();
}

StatusOr<gpu::InfeedBuffer*> GpuTransferManager::TransferBufferToInfeedInternal(
    se::StreamExecutor* executor, int64 size, const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %lld bytes", size);
  }

  if (size == 0) {
    return InvalidArgument("Infeed shape needs 0 bytes");
  }

  gpu::InfeedManager* infeed_manager = gpu::GetOrCreateInfeedManager();
  se::Stream* stream = infeed_manager->GetStream(executor);
  if (stream == nullptr) {
    return InternalError("Failed to obtain a stream");
  }

  gpu::InfeedBuffer* buffer = new gpu::InfeedBuffer(executor, size);
  stream->ThenMemcpy(buffer->device_memory(), source, size);

  VLOG(2) << "Queued infeed data on stream " << stream;

  return buffer;
}

}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateNVGpuTransferManager() {
  return xla::MakeUnique<xla::GpuTransferManager>(
      stream_executor::cuda::kCudaPlatformId);
}

static std::unique_ptr<xla::TransferManager> CreateAMDGpuTransferManager() {
  return xla::MakeUnique<xla::GpuTransferManager>(
      stream_executor::rocm::kROCmPlatformId);
}

static bool InitModule() {
  // XXX figure out how to support both AMDGPU and NVPTX at the same time
  xla::TransferManager::RegisterTransferManager(
      stream_executor::rocm::kROCmPlatformId, &CreateAMDGpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/neg.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/neg.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace neg {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteInt8:
      // using EQ attempts to cast to int via the %d format specifier and gives
      // incorrect value and since some hardware platforms do not support float
      // formatting by default, I did a direct equality check instead
      TF_LITE_ENSURE(context, input->params.scale == output->params.scale);
      reference_integer_ops::Negate(
          GetTensorShape(input), GetTensorData<int8_t>(input),
          input->params.zero_point, GetTensorShape(output),
          GetTensorData<int8_t>(output), output->params.zero_point);
      break;

    case kTfLiteFloat32:
      reference_ops::Negate(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      break;

    default:
      context->ReportError(
          context, "Neg only currently supports float32 and int8, got %d.",
          input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace neg

TfLiteRegistration* Register_NEG() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 /*prepare=*/nullptr, neg::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

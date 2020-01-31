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
#include <initializer_list>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// TODO(See TfLiteSqueezeParams): We can't have dynamic data, at least not
// yet. For now we will fix the maximum possible number of dimensions.
constexpr int max_num_dims = 8;

template <typename T = float, TfLiteType tensor_input_type = kTfLiteFloat32>
void TestSqueeze(std::initializer_list<int> input_dims_data,
                 std::initializer_list<T> input_data,
                 std::initializer_list<int> squeeze_dims,
                 std::initializer_list<int> expected_output_dims_data,
                 std::initializer_list<T> expected_output,
                 uint8_t* output_data_raw) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* initial_output_dims =
      IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* expected_output_dims =
      IntArrayFromInitializer(expected_output_dims_data);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  T* initial_output_data = reinterpret_cast<T*>(output_data_raw);
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor<T, tensor_input_type>(input_data, input_dims,
                                         "input_tensor"),
      CreateTensor<T, tensor_input_type>(initial_output_data,
                                         initial_output_dims, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SQUEEZE, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  // TODO(See TfLiteSqueezeParams): We can't have dynamic data, at least not
  // yet. Thus we allocate space for outputs_dims_array in the temporaries
  // buffer.
  int outputs_dims_array_data[max_num_dims + 1] = {
      max_num_dims, 0, 0, 0, 0, 0, 0, 0, 0};
  TfLiteIntArray* outputs_dims_array =
      IntArrayFromInts(outputs_dims_array_data);

  TfLiteSqueezeParams params;
  TF_LITE_MICRO_EXPECT_LE(squeeze_dims.size(), max_num_dims);
  params.num_squeeze_dims = squeeze_dims.size();
  for (int i = 0; i < params.num_squeeze_dims; ++i) {
    params.squeeze_dims[i] = squeeze_dims.begin()[i];
  }

  // Run op
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = outputs_dims_array;
  node.user_data = nullptr;
  node.builtin_data = &params;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  // Check the squeezed dimensions are as expected
  constexpr int outputs_index = 1;
  TfLiteIntArray* output_dims = tensors[outputs_index].dims;

  TF_LITE_MICRO_EXPECT_EQ(expected_output_dims->size, output_dims->size);
  for (int i = 0; i < output_dims->size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_dims->data[i],
                            output_dims->data[i]);
  }

  // Check that squeeze does not mutate the data
  const T* output_data = GetTensorData<T>(&tensors[outputs_index]);
  int flat_size = 1;
  for (int i = 0; i < output_dims->size; ++i) {
    flat_size *= output_dims->data[i];
  }
  for (int i = 0; i < flat_size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output.begin()[i], output_data[i],
                              1e-5f);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SqueezeAll) {
  uint8_t output_data_buffer[8 * 4];
  constexpr int num_input_dims = 3;
  constexpr int num_expected_output_dims = 1;

  tflite::testing::TestSqueeze(
      {num_input_dims, 1, 8, 1},                    // input_dims
      {1, 2, 3, 4, 5, 6, 7, 8},                     // input_data
      {},                                           // squeeze_dims
      {num_expected_output_dims, 8},                // expected_output_dims
      {1, 2, 3, 4, 5, 6, 7, 8}, output_data_buffer  // expected_output
  );
}

TF_LITE_MICRO_TEST(SqueezeSelectedAxis) {
  uint8_t output_data_buffer[8 * 4];
  constexpr int num_input_dims = 3;
  constexpr int num_expected_output_dims = 2;

  tflite::testing::TestSqueeze(
      {num_input_dims, 1, 8, 1},                    // input_dims
      {1, 2, 3, 4, 5, 6, 7, 8},                     // input_data
      {2},                                          // squeeze_dims
      {num_expected_output_dims, 1, 8},             // expected_output_dims
      {1, 2, 3, 4, 5, 6, 7, 8}, output_data_buffer  // expected_output
  );
}

TF_LITE_MICRO_TEST(SqueezeNegativeAxis) {
  uint8_t output_data_buffer[8 * 4];
  constexpr int num_input_dims = 3;
  constexpr int num_expected_output_dims = 1;

  tflite::testing::TestSqueeze(
      {num_input_dims, 1, 8, 1},                    // input_dims
      {1, 2, 3, 4, 5, 6, 7, 8},                     // input_data
      {-1, 0},                                      // squeeze_dims
      {num_expected_output_dims, 8},                // expected_output_dims
      {1, 2, 3, 4, 5, 6, 7, 8}, output_data_buffer  // expected_output
  );
}

TF_LITE_MICRO_TEST(SqueezeRepetitiveAxis) {
  uint8_t output_data_buffer[8 * 4];
  constexpr int num_input_dims = 3;
  constexpr int num_expected_output_dims = 1;

  tflite::testing::TestSqueeze(
      {num_input_dims, 1, 8, 1},                    // input_dims
      {1, 2, 3, 4, 5, 6, 7, 8},                     // input_data
      {2, 2, 0, -1, -3},                            // squeeze_dims
      {num_expected_output_dims, 8},                // expected_output_dims
      {1, 2, 3, 4, 5, 6, 7, 8}, output_data_buffer  // expected_output
  );
}

TF_LITE_MICRO_TEST(SqueezeAllDims) {
  uint8_t output_data_buffer[1 * 4];
  constexpr int num_input_dims = 7;
  constexpr int num_expected_output_dims = 0;

  tflite::testing::TestSqueeze(
      {num_input_dims, 1, 1, 1, 1, 1, 1, 1},  // input_dims
      {3},                                    // input_data
      {},                                     // squeeze_dims
      {num_expected_output_dims},             // expected_output_dims
      {3}, output_data_buffer                 // expected_output
  );
}

TF_LITE_MICRO_TESTS_END

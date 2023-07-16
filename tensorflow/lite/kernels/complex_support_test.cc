/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <complex>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

template <typename T>
class ComplexOpModel : public SingleOpModel {
 public:
  ComplexOpModel(const TensorData& real_input, const TensorData& imag_input,
                 const TensorData& output) {
    real_input_ = AddInput(real_input);
    imag_input_ = AddInput(imag_input);

    output_ = AddOutput(output);

    const std::vector<uint8_t> custom_option;
    SetBuiltinOp(BuiltinOperator_COMPLEX, BuiltinOptions_NONE, 0);

    BuildInterpreter({GetShape(real_input_), GetShape(imag_input_)});
  }

  int real_input() { return real_input_; }
  int imag_input() { return imag_input_; }

  std::vector<std::complex<T>> GetOutput() { return ExtractVector<std::complex<T>>(output_); }

 private:
  int real_input_;
  int imag_input_;
  int output_;
};


TEST(ComplexOpTest, SimpleFloatTest) {
  ComplexOpModel<float> m({TensorType_FLOAT32, {1, 8}},
                          {TensorType_FLOAT32, {1, 8}},
                          {TensorType_COMPLEX64, {}});

  m.PopulateTensor<float>(
      m.real_input(), {{75.0f, -6.0f, 9.0f, -10.0f, -3.0f, -6.0f, 0.0f, 22.1f}});
  m.PopulateTensor<float>(m.imag_input(),
                          {{0.0f, -1.0f, 0.0f, 5.0f, 2.0f, 11.0f, 0.0f, 33.3f}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray(ArrayComplex64Near({{75.0f, 0.0f},
                                                        {-6.0f, -1.0f},
                                                        {9.0f, 0.0f},
                                                        {-10.0f, 5.0f},
                                                        {-3.0f, 2.0f},
                                                        {-6.0f, 11.0f},
                                                        {0.0f, 0.0f},
                                                        {22.1f, 33.3f}})));
}

TEST(ComplexOpTest, SimpleDoubleTest) {
  ComplexOpModel<double> m({TensorType_FLOAT64, {1, 8}},
                           {TensorType_FLOAT64, {1, 8}},
                           {TensorType_COMPLEX128, {}});

  m.PopulateTensor<double>(m.real_input(),
                          {{75.0, -6.0, 9.0, -10.0, -3.0, -6.0, 0.0, 22.1}});
  m.PopulateTensor<double>(m.imag_input(),
                          {{0.0, -1.0, 0.0, 5.0, 2.0, 11.0, 0.0, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray(ArrayComplex64Near({{75.0, 0.0},
                                                        {-6.0, -1.0},
                                                        {9.0, 0.0},
                                                        {-10.0, 5.0},
                                                        {-3.0, 2.0},
                                                        {-6.0, 11.0},
                                                        {0.0, 0.0},
                                                        {22.1, 33.3}})));
}

template <typename T>
class RealOpModel : public SingleOpModel {
 public:
  RealOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);

    output_ = AddOutput(output);

    const std::vector<uint8_t> custom_option;
    SetBuiltinOp(BuiltinOperator_REAL, BuiltinOptions_NONE, 0);

    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int output_;
};

TEST(RealOpTest, SimpleFloatTest) {
  RealOpModel<float> m({TensorType_COMPLEX64, {2, 4}},
                       {TensorType_FLOAT32, {}});

  m.PopulateTensor<std::complex<float>>(m.input(), {{75, 0},
                                                    {-6, -1},
                                                    {9, 0},
                                                    {-10, 5},
                                                    {-3, 2},
                                                    {-6, 11},
                                                    {0, 0},
                                                    {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {75, -6, 9, -10, -3, -6, 0, 22.1f})));
}

TEST(RealOpTest, SimpleDoubleTest) {
  RealOpModel<double> m({TensorType_COMPLEX128, {2, 4}},
                        {TensorType_FLOAT64, {}});

  m.PopulateTensor<std::complex<double>>(m.input(), {{75, 0},
                                                     {-6, -1},
                                                     {9, 0},
                                                     {-10, 5},
                                                     {-3, 2},
                                                     {-6, 11},
                                                     {0, 0},
                                                     {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {75, -6, 9, -10, -3, -6, 0, 22.1f})));
}

template <typename T>
class ImagOpModel : public SingleOpModel {
 public:
  ImagOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);

    output_ = AddOutput(output);

    const std::vector<uint8_t> custom_option;
    SetBuiltinOp(BuiltinOperator_IMAG, BuiltinOptions_NONE, 0);

    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int output_;
};

TEST(ImagOpTest, SimpleFloatTest) {
  ImagOpModel<float> m({TensorType_COMPLEX64, {2, 4}},
                       {TensorType_FLOAT32, {}});

  m.PopulateTensor<std::complex<float>>(m.input(), {{75, 7},
                                                    {-6, -1},
                                                    {9, 3.5},
                                                    {-10, 5},
                                                    {-3, 2},
                                                    {-6, 11},
                                                    {0, 0},
                                                    {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {7, -1, 3.5f, 5, 2, 11, 0, 33.3f})));
}

TEST(ImagOpTest, SimpleDoubleTest) {
  ImagOpModel<double> m({TensorType_COMPLEX128, {2, 4}},
                        {TensorType_FLOAT64, {}});

  m.PopulateTensor<std::complex<double>>(m.input(), {{75, 7},
                                                     {-6, -1},
                                                     {9, 3.5},
                                                     {-10, 5},
                                                     {-3, 2},
                                                     {-6, 11},
                                                     {0, 0},
                                                     {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {7, -1, 3.5f, 5, 2, 11, 0, 33.3f})));
}

template <typename T>
class ComplexAbsOpModel : public SingleOpModel {
 public:
  ComplexAbsOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);

    output_ = AddOutput(output);

    const std::vector<uint8_t> custom_option;
    SetBuiltinOp(BuiltinOperator_COMPLEX_ABS, BuiltinOptions_NONE, 0);

    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(ComplexAbsOpTest, IncompatibleType64Test) {
  EXPECT_DEATH_IF_SUPPORTED(
      ComplexAbsOpModel<float> m({TensorType_COMPLEX64, {2, 4}},
                                 {TensorType_FLOAT64, {}}),
      "output->type != kTfLiteFloat32");
}

TEST(ComplexAbsOpTest, IncompatibleType128Test) {
  EXPECT_DEATH_IF_SUPPORTED(
      ComplexAbsOpModel<float> m({TensorType_COMPLEX128, {2, 4}},
                                 {TensorType_FLOAT32, {}}),
      "output->type != kTfLiteFloat64");
}

TEST(ComplexAbsOpTest, SimpleFloatTest) {
  ComplexAbsOpModel<float> m({TensorType_COMPLEX64, {2, 4}},
                             {TensorType_FLOAT32, {}});

  m.PopulateTensor<std::complex<float>>(m.input(), {{75, 7},
                                                    {-6, -1},
                                                    {9, 3.5},
                                                    {-10, 5},
                                                    {-3, 2},
                                                    {-6, 11},
                                                    {0, 0},
                                                    {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), testing::ElementsAre(2, 4));
  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {75.32596f, 6.0827627f, 9.656604f, 11.18034f,
                                  3.6055512f, 12.529964f, 0.f, 39.966236f})));
}

TEST(ComplexAbsOpTest, SimpleDoubleTest) {
  ComplexAbsOpModel<double> m({TensorType_COMPLEX128, {2, 4}},
                              {TensorType_FLOAT64, {}});

  m.PopulateTensor<std::complex<double>>(m.input(), {{75, 7},
                                                     {-6, -1},
                                                     {9, 3.5},
                                                     {-10, 5},
                                                     {-3, 2},
                                                     {-6, 11},
                                                     {0, 0},
                                                     {22.1, 33.3}});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), testing::ElementsAre(2, 4));
  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(ArrayFloatNear(
                                 {75.32596f, 6.0827627f, 9.656604f, 11.18034f,
                                  3.6055512f, 12.529964f, 0.f, 39.966236f})));
}

}  // namespace
}  // namespace tflite

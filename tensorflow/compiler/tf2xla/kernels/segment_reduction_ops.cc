/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

class UnsortedSegmentReduce : public XlaOpKernel {
 public:
  explicit UnsortedSegmentReduce(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &type_));
  }

  // The initial value to initialize elements of the output to.
  virtual xla::XlaOp InitialValue(xla::XlaBuilder* builder) = 0;

  // A function to combine two scalars with the same index (e.g., sum).
  virtual xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) = 0;

  void Compile(XlaOpKernelContext* ctx) override {
    // output = unsorted_segment_sum(data, indices, num_segments)
    // Compute a tensor such that:
    //    output[i] = sum over {j where indices[j] == i} of data[j]
    //    output[i] == 0 if i does not appear in indices
    //
    // Contrast with segment_sum(), which assumes indices are sorted and that
    // max(indices)+1 is the desired size of the output.
    //
    // The returned output tensor has the same type as data, and the same shape
    // as data with the first indices.rank dimensions are replaced
    // by a single dimension with size num_segments.
    auto data = ctx->Input(0);
    TensorShape data_shape = ctx->InputShape(0);

    auto indices = ctx->Input(1);
    TensorShape indices_shape = ctx->InputShape(1);

    int64 num_segments;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &num_segments));

    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(type_string(),
                                        " requires that indices' rank be"
                                        " less than or equal to data's rank."));
    // Validate that indices.shape is a prefix of data.shape.
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(
          ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
          errors::InvalidArgument(type_string(),
                                  " requires indices shape to be prefix"
                                  " of data_shape, but dimension ",
                                  d, " differs ", data_shape.dim_size(d),
                                  " vs. ", indices_shape.dim_size(d)));
    }
    xla::XlaBuilder* builder = ctx->builder();
    TensorShape buffer_shape = data_shape;
    buffer_shape.RemoveDimRange(0, indices_shape.dims());
    buffer_shape.InsertDim(0, num_segments);
    auto buffer =
        xla::Broadcast(InitialValue(builder), buffer_shape.dim_sizes());

    auto combiner = [this](xla::XlaOp a, xla::XlaOp b,
                           xla::XlaBuilder* builder) { return Combine(a, b); };

    auto result = XlaScatter(buffer, /*updates=*/data, indices,
                             /*indices_are_vectors=*/false, combiner, builder);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.ValueOrDie());
  }

 protected:
  xla::PrimitiveType type_;
};

class UnsortedSegmentSum : public UnsortedSegmentReduce {
 public:
  explicit UnsortedSegmentSum(OpKernelConstruction* ctx)
      : UnsortedSegmentReduce(ctx) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a + b; };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentSum").CompileTimeConstInput("num_segments"),
    UnsortedSegmentSum);

class UnsortedSegmentProd : public UnsortedSegmentReduce {
 public:
  explicit UnsortedSegmentProd(OpKernelConstruction* ctx)
      : UnsortedSegmentReduce(ctx) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::One(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a * b; };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentProd").CompileTimeConstInput("num_segments"),
    UnsortedSegmentProd);

class UnsortedSegmentMin : public UnsortedSegmentReduce {
 public:
  explicit UnsortedSegmentMin(OpKernelConstruction* ctx)
      : UnsortedSegmentReduce(ctx) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MaxFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Min(a, b);
  };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentMin").CompileTimeConstInput("num_segments"),
    UnsortedSegmentMin);

class UnsortedSegmentMax : public UnsortedSegmentReduce {
 public:
  explicit UnsortedSegmentMax(OpKernelConstruction* ctx)
      : UnsortedSegmentReduce(ctx) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MinFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Max(a, b);
  };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentMax").CompileTimeConstInput("num_segments"),
    UnsortedSegmentMax);

}  // namespace
}  // namespace tensorflow

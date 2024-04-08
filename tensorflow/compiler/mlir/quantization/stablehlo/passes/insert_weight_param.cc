/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_INSERTWEIGHTPARAMPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Inserts quantization parameters of weights for weight-only quantization and
// dynamic range quantization of `stablehlo.convolution` and
// `stablehlo.dot_general`.
class InsertWeightParamPass
    : public impl::InsertWeightParamPassBase<InsertWeightParamPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertWeightParamPass)

  using impl::InsertWeightParamPassBase<
      InsertWeightParamPass>::InsertWeightParamPassBase;

  explicit InsertWeightParamPass(
      const bool enable_per_channel_quantized_weight) {
    enable_per_channel_quantized_weight_ = enable_per_channel_quantized_weight;
  }

 private:
  void runOnOperation() override;
};

// Inserts quantization parameters for weights for hybrid quantization of
// `stablehlo.convolution` and `stablehlo.dot_general`.
class InsertWeightParamPattern
    : public OpTraitRewritePattern<OpTrait::ConstantLike> {
 public:
  using OpTraitRewritePattern<OpTrait::ConstantLike>::OpTraitRewritePattern;

  explicit InsertWeightParamPattern(
      MLIRContext* context, const bool enable_per_channel_quantized_weight)
      : OpTraitRewritePattern<OpTrait::ConstantLike>(context),
        enable_per_channel_quantized_weight_(
            enable_per_channel_quantized_weight) {}

  LogicalResult match(Operation* op) const override {
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto type = op->getResult(0).getType().cast<TensorType>();
    if (!type || !type.getElementType().isF32()) {
      return failure();
    }
    return success(op->hasOneUse() &&
                   IsWeightQuantizableFunction(*op->getUses().begin()));
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    Operation* quantizable_op = *op->getUsers().begin();
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) {
      return;
    }
    Type weight_type;
    if (enable_per_channel_quantized_weight_) {
      auto call_op = cast<TF::XlaCallModuleOp>(quantizable_op);
      FailureOr<int64_t> quant_dim = GetQuantizationDimension(call_op);
      if (failed(quant_dim)) {
        op->emitError("Failed to get quantization dimension for weight.");
        return;
      }
      weight_type = quant::GetUniformQuantizedPerAxisTypeForWeight(
          attr, *quant_dim, /*symmetric=*/false, /*num_bits=*/8,
          /*is_signed=*/true,
          /*narrow_range=*/false, /*legacy_float_scale=*/false);
    } else {
      weight_type = quant::GetUniformQuantizedTypeForWeight(
          attr, /*symmetric=*/true, /*num_bits=*/8,
          /*is_signed=*/true,
          /*narrow_range=*/true, /*legacy_float_scale=*/false);
    }

    auto quant_type = weight_type.template dyn_cast<quant::QuantizedType>();
    if (!quant_type) {
      op->emitError(
          "Failed to get weight quantization parameters for weight-only "
          "quantization.");
    }

    const Type expressed_type = op->getResult(0).getType();
    const Type quantized_type =
        quant_type.castFromExpressedType(expressed_type);

    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<quantfork::QuantizeCastOp>(
        op->getLoc(), quantized_type, op->getResult(0));
    auto dq = rewriter.create<quantfork::DequantizeCastOp>(op->getLoc(),
                                                           expressed_type, q);
    quantizable_op->setOperand(1, dq.getResult());
  }

 private:
  // Checks if the operand is second operand of `tf.XlaCallModule` op for
  // `stablehlo.convolution` or `stablehlo.dot_general` with fully_quantizable
  // trait.
  static bool IsWeightQuantizableFunction(OpOperand& operand) {
    if (operand.getOperandNumber() != 1) {
      return false;
    }
    Operation* user = operand.getOwner();
    if (isa<TF::XlaCallModuleOp>(user)) {
      auto call_op = cast<TF::XlaCallModuleOp>(user);
      const StringRef function_name = GetEntryFunctionName(call_op);
      const bool is_conv_or_dot = function_name.contains("conv") ||
                                  function_name.contains("dot_general");
      const bool has_quant_trait = HasQuantizableTrait(call_op);
      return is_conv_or_dot && has_quant_trait;
    }
    return false;
  }

  // Determines quantization dimension of weights for given `tf.XlaCallModule`
  // op. For convolution, returns output feature dimension of the kernel. For
  // dot_general, returns the first non-contracting dimension, non-batching
  // dimension. If no such dimension exists, raises error.
  static FailureOr<int64_t> GetQuantizationDimension(TF::XlaCallModuleOp op) {
    const StringRef function_name = GetEntryFunctionName(op);
    const auto module_op = op->getParentOfType<ModuleOp>();
    const SymbolTable symbol_table(module_op);
    func::FuncOp func = symbol_table.lookup<func::FuncOp>(function_name);

    if (function_name.contains("conv")) {
      auto conv = *(func.getOps<mlir::stablehlo::ConvolutionOp>().begin());
      return conv.getDimensionNumbers().getKernelOutputFeatureDimension();
    } else if (function_name.contains("dot_general")) {
      auto dot = *(func.getOps<mlir::stablehlo::DotGeneralOp>().begin());
      ::mlir::stablehlo::DotDimensionNumbersAttr dimension_numbers =
          dot.getDotDimensionNumbers();
      auto rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      auto rhs_batching_dims = dimension_numbers.getRhsBatchingDimensions();
      auto rank = dot.getRhs().getType().cast<TensorType>().getRank();
      for (int i = 0; i < rank; ++i) {
        // Return the first non-contracting, non-batching dimension of rhs.
        if (llvm::find(rhs_contracting_dims, i) == rhs_contracting_dims.end() &&
            llvm::find(rhs_batching_dims, i) == rhs_batching_dims.end()) {
          return i;
        }
      }
      op->emitError(
          "dot_general op does not have non-contracting, non-batching "
          "dimension.");
      return failure();
    }
    op->emitError(
        "Weight-only quantization only applies to convolution and "
        "dot_general.");
    return failure();
  }

  const bool enable_per_channel_quantized_weight_;
};

void InsertWeightParamPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* context = func.getContext();
  RewritePatternSet patterns(context);

  patterns.add<InsertWeightParamPattern>(context,
                                         enable_per_channel_quantized_weight_);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo

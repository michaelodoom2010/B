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

// See docs in ../ops/linalg_ops.cc.
#include <vector>


#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"

namespace tensorflow {

static const char kErrMsg[] =
    "LU decomposition was not successful. The input might not be valid.";

template <class Scalar>
class LuOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit LuOp(OpKernelConstruction* context) : Base(context) {}


  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64 m = input_matrix_shapes[0].dim_size(0);  
    // only square matrix is supported for now.
    return TensorShapes({TensorShape({m, m}), 
                         TensorShape({m, m}),
                         TensorShape({m})}); // 1, m
  }


  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      return;
    }

    // Perform the actual LU decomposition.
    Eigen::PartialPivLU<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        lu_decomposition(input);

    //OP_REQUIRES(context, lu_decomposition.isInvertible() == true,
    //            errors::InvalidArgument(kErrMsg));
    // Output the lower triangular in a dense form.
    outputs->at(0) =
        lu_decomposition.matrixLU().template triangularView<Eigen::UnitLower>();
    outputs->at(1) =
        lu_decomposition.matrixLU().template triangularView<Eigen::Upper>();        
    //outputs->at(2) = lu_decomposition.permutationP();//.indices().data();    
    //Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  perm(input.rows());
    Eigen::VectorXd perm(input.rows()); 
    outputs->at(2) = perm.cast<Scalar>();
    
    auto & indices = lu_decomposition.permutationP().indices();    
    for(int i = 0; i < indices.size(); i++){
        perm(i) = indices(i);        
    }
    using namespace std;
    cout<<perm<<endl;   
    //lu_decomposition.permutationP();//.indices().data();    
    //outputs->at(2) = lu_decomposition.permutationP().indices().array();        
    //using namespace std;
    //cout<<"permutation matrix"<<endl;
    //cout<<lu_decomposition.permutationP().indices()<<endl;//.cast<int>().array();
    //int n = input.rows();
    //Tensor perm_vec;//(n, 1);
    //outputs->at(2) = perm_vec;    
    //Eigen::ArrayXi perm = lu_decomposition.permutationP().indices().cast<int>().array();
    //for(const auto & it:perm) cout<<it<<" ";
  }
};

REGISTER_LINALG_OP("Lu", (LuOp<float>), float);
REGISTER_LINALG_OP("Lu", (LuOp<double>), double);
REGISTER_LINALG_OP("Lu", (LuOp<complex64>), complex64);
REGISTER_LINALG_OP("Lu", (LuOp<complex128>), complex128);
}  // namespace tensorflow

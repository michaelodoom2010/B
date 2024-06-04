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

#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"

#include <string>

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {

namespace {

TEST(GetNextAllowedBatchSizeTest, PaddingDisallowed) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {2, 4, 8}, true), 3);
}

TEST(GetNextAllowedBatchSizeTest, EmptyAllowedBatchSizes) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {}, false), 3);
}

TEST(GetNextAllowedBatchSizeTest, NextAllowedBatchSizeFound) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {2, 4, 8}, false), 4);
}

TEST(GetNextAllowedBatchSizeTest, AlreadyAllowedBatchSize) {
  EXPECT_EQ(GetNextAllowedBatchSize(2, {2, 4, 8}, false), 2);
}

TEST(GetNextAllowedBatchSizeTest, GreaterThanAllowedBatchSize) {
  EXPECT_EQ(GetNextAllowedBatchSize(10, {2, 4, 8}, false), 10);
}

TEST(BatchPaddingPolicyTest, AbslParseFlag) {
  std::string error;
  BatchPaddingPolicy policy;

  EXPECT_TRUE(AbslParseFlag("PAD_UP", &policy, &error));
  EXPECT_EQ(policy, BatchPaddingPolicy::kPadUp);
  EXPECT_EQ(error, "");

  EXPECT_TRUE(AbslParseFlag("BATCH_DOWN", &policy, &error));
  EXPECT_EQ(policy, BatchPaddingPolicy::kBatchDown);
  EXPECT_EQ(error, "");

  EXPECT_TRUE(AbslParseFlag("MINIMIZE_TPU_COST_PER_REQUEST", &policy, &error));
  EXPECT_EQ(policy, BatchPaddingPolicy::kMinimizeTpuCostPerRequest);
  EXPECT_EQ(error, "");

  EXPECT_FALSE(AbslParseFlag("cucumber", &policy, &error));
  EXPECT_NE(error, "");
}

TEST(BatchPaddingPolicyTest, AbslUnparseFlag) {
  EXPECT_EQ(AbslUnparseFlag(BatchPaddingPolicy::kPadUp), "PAD_UP");
  EXPECT_EQ(AbslUnparseFlag(BatchPaddingPolicy::kBatchDown), "BATCH_DOWN");
  EXPECT_EQ(AbslUnparseFlag(BatchPaddingPolicy::kMinimizeTpuCostPerRequest),
            "MINIMIZE_TPU_COST_PER_REQUEST");
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow

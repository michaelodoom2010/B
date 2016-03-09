/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/ops/compat/op_compatibility_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(BackwardsCompatibilityTest, IsCompatible) {
  OpCompatibilityLib compatibility("tensorflow/core/ops");

  // Read ops.pbtxt and compare with the full versions of all ops.
  Env* env = Env::Default();
  bool ops_changed = false;
  {
    const string& ops_file = compatibility.ops_file();
    printf("Reading ops from %s...\n", ops_file.c_str());
    string ops_str;
    TF_ASSERT_OK(ReadFileToString(env, ops_file, &ops_str));
    ops_changed = ops_str != compatibility.OpsString();
  }

  int changed_ops = 0;
  int added_ops = 0;
  TF_ASSERT_OK(
      compatibility.ValidateCompatible(env, &changed_ops, &added_ops, nullptr));
  printf("%d changed ops\n%d added ops\n", changed_ops, added_ops);

  if (ops_changed || changed_ops + added_ops > 0) {
    if (changed_ops + added_ops == 0) {
      printf("Only Op documentation changed.\n");
    }
    ADD_FAILURE()
        << "Please run:\n"
           "  tensorflow/core/ops/compat/update_ops <core/ops directory>\n"
           "to update the checked-in list of all ops.\n";
  }
}

}  // namespace
}  // namespace tensorflow

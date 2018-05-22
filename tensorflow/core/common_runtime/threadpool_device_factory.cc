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

// Register a factory that provides CPU devices.

#include "tensorflow/core/common_runtime/numa_device.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"

#include <vector>
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"

namespace tensorflow {

// TODO(zhifengc/tucker): Figure out the bytes of available RAM.
class ThreadPoolDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    // TODO(zhifengc/tucker): Figure out the number of available CPUs
    // and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }

    int32 intra_op_parallelism_threads =
        options.config.intra_op_parallelism_threads();
    std::vector<int>  proc_set[1];
    /*if(intra_op_parallelism_threads == 56) {
     for(int i=0; i<intra_op_parallelism_threads; i++){
        proc_set[0].push_back(i+56);
      }
      std::cout << "Create CPU device, bound to cores:\n";
      for(int i=0; i<proc_set[0].size(); i++)
        std::cout << proc_set[0][i] << " ";
      std::cout << "\n";
      for (int i = 0; i < n; i++) {
        string name = strings::StrCat(name_prefix, "/device:CPU:", i);
        devices->push_back(new NumaDevice(
            options, name, Bytes(256 << 20), DeviceLocality(), new MklCPUAllocator(), proc_set[i]));
      }
    }
    else */{
      for (int i = 0; i < n; i++) {
        string name = strings::StrCat(name_prefix, "/device:CPU:", i);
        devices->push_back(new ThreadPoolDevice(
            options, name, Bytes(256 << 20), DeviceLocality(), new MklCPUAllocator()));
      }
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory, 60);

}  // namespace tensorflow

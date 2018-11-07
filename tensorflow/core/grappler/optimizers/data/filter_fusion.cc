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

#include "tensorflow/core/grappler/optimizers/data/filter_fusion.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeFusedFilterNode(const NodeDef& first_filter_node,
                            const NodeDef& second_filter_node,
                            const FunctionDef& fused_function,
                            MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_filter", graph->graph(),
                                      &fused_node);

  fused_node.set_op("FilterDataset");
  fused_node.add_input(first_filter_node.input(0));

  auto attr = first_filter_node.attr().at("predicate");
  *attr.mutable_func()->mutable_name() = fused_function.signature().name();
  (*fused_node.mutable_attr())["predicate"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", first_filter_node, &fused_node);

  for (auto key : {"output_shapes", "output_types"})
    graph_utils::CopyAttribute(key, second_filter_node, &fused_node);

  return fused_node;
}

}  // namespace

Status FilterFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                              GraphDef* output) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             output->library());

  auto get_filter_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == "FilterDataset") return &node;
    return nullptr;
  };

  auto get_fused_predicate =
      [&](const NodeDef* first_filter_node,
          const NodeDef* second_filter_node) -> FunctionDef* {
    const auto& parent_fun = first_filter_node->attr().at("predicate");
    const FunctionDef* first_func =
        function_library.Find(parent_fun.func().name());
    const auto& fun = second_filter_node->attr().at("predicate");
    const FunctionDef* second_func = function_library.Find(fun.func().name());

    if (!fusion_utils::HasSameSignature(first_func->signature(),
                                        second_func->signature())) {
      VLOG(1) << "Can't fuse Filters because they have different signature\n";
      return nullptr;
    }

    return fusion_utils::FuseFunctions(
        *first_func, *second_func, "fused_predicate",
        fusion_utils::SameSignature, fusion_utils::SameInput,
        fusion_utils::LazyConjunctionOutput, fusion_utils::LazyConjunctionNodes,
        output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* second_filter_node = get_filter_node(node);
    if (!second_filter_node) continue;

    const NodeDef* first_filter_node =
        get_filter_node(*graph_utils::GetInputNode(*second_filter_node, graph));
    if (!first_filter_node) continue;

    const auto* fused_predicate =
        get_fused_predicate(first_filter_node, second_filter_node);
    if (!fused_predicate) continue;
    const auto* fused_filter_node = graph.AddNode(MakeFusedFilterNode(
        *first_filter_node, *second_filter_node, *fused_predicate, &graph));

    graph.UpdateFanouts(second_filter_node->name(), fused_filter_node->name());

    // TODO(prazek): we should run some optimizations on the fused filter
    // functions, or make sure that optimization passes run after filter
    // fusion.
    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_predicate));
    // TODO(b/116285210): we could also remove map functions from library if
    // they are not used anymore.
    nodes_to_delete.insert(first_filter_node->name());
    nodes_to_delete.insert(second_filter_node->name());
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void FilterFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                            const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(FilterFusion, "filter_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

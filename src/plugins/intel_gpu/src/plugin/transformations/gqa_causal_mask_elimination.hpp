// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Eliminates the WebNN GQA causal mask subgraph from IndirectSDPA inputs.
///
/// WebNN's ONNX GQA decomposition generates a dynamic causal mask subgraph per decode step:
///   attention_mask → ReduceSum → Add → Select(where) → Add → Broadcast → Transpose
///   value_int_one → Broadcast → CumSum(exclusive)
///   CumSum < Transpose → Less → Clamp → Select(where) → Squeeze → IndirectSDPA[mask]
///
/// This is equivalent to the standard causal attention mask that IndirectSDPA's `is_causal=true`
/// implements natively in the SDPA kernel. The pass removes the mask input and sets is_causal=true.
class WebNNGQACausalMaskEliminationSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WebNNGQACausalMaskEliminationSDPA");
    WebNNGQACausalMaskEliminationSDPA();
};

class WebNNGQACausalMaskElimination : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("WebNNGQACausalMaskElimination");
    WebNNGQACausalMaskElimination();
};

}  // namespace ov::intel_gpu

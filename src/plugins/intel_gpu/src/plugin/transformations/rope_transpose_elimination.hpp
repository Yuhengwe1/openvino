// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Eliminates redundant Transpose-Reshape-Transpose chains after WebNN RoPE fusion.
///
/// Pattern: RoPE(input_trans0213) -> Transpose(0,2,1,3) -> Reshape(s) -> Transpose(0,2,1,3)
/// Becomes: RoPE -> [consumers of outer Transpose]
///
/// The two Transpose(0,2,1,3) cancel (net identity); intermediate Reshapes are identity.
class WebNNRoPETransposeEliminationKV : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WebNNRoPETransposeEliminationKV");
    WebNNRoPETransposeEliminationKV();
};

class WebNNRoPETransposeElimination : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("WebNNRoPETransposeElimination");
    WebNNRoPETransposeElimination();
};

}  // namespace ov::intel_gpu

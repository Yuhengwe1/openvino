// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Matches WebNN/ORT RotaryEmbedding decomposition pattern (5D intermediate)
/// and replaces it with the fused ov::op::internal::RoPE node.
///
/// Pattern:
///   Reshape([B,L,H*D] -> [B,L,H,2,half_D])
///   -> Multiply(x, cos)  [mul_cos]
///   -> VariadicSplit(x, axis=3, [1,1])  [split into halves]
///   -> Multiply(half1, -1)  [negate]
///   -> Concat([neg, half0], axis=3)  [rotate_half]
///   -> Multiply(rotate_half, sin)  [mul_sin]
///   -> Add(mul_cos, mul_sin)  [result]
///   -> Reshape([B,L,H,2,half_D] -> [B,L,H,D])
class WebNNRoPEFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WebNNRoPEFusionMatcher");
    WebNNRoPEFusionMatcher();
};

class WebNNRoPEFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("WebNNRoPEFusion");
    WebNNRoPEFusion();
};

/// After WebNNRoPEFusion creates RoPE(data, gathered_cos, gathered_sin),
/// this pass detects that cos/sin inputs come from Gather(table, position_ids, axis)
/// and folds the Gather into the RoPE kernel (using gather_position_arg_id=3).
/// Eliminates 2 GPU Gather kernel dispatches per RoPE (44 total across Q+K × 22 layers).
class WebNNRoPEGatherFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WebNNRoPEGatherFusionMatcher");
    WebNNRoPEGatherFusionMatcher();
};

class WebNNRoPEGatherFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("WebNNRoPEGatherFusion");
    WebNNRoPEGatherFusion();
};

}  // namespace ov::intel_gpu

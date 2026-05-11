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

}  // namespace ov::intel_gpu

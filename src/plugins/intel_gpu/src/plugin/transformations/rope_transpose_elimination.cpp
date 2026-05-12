// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_transpose_elimination.hpp"

#include <memory>
#include <vector>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

namespace {

bool is_transpose_0213(const std::shared_ptr<ov::op::v1::Transpose>& transpose) {
    auto order_node = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!order_node)
        return false;
    auto order = order_node->cast_vector<int64_t>();
    return order == std::vector<int64_t>{0, 2, 1, 3};
}

}  // namespace

WebNNRoPETransposeEliminationKV::WebNNRoPETransposeEliminationKV() {
    using namespace ov::pass::pattern;

    // Pattern: RoPE -> Transpose(0213) -> Reshape -> [optional Reshape] -> Transpose(0213)
    // The two Transpose(0,2,1,3) cancel each other (net identity).
    auto rope_m = wrap_type<ov::op::internal::RoPE>();
    auto transpose_inner_m = wrap_type<ov::op::v1::Transpose>({rope_m, any_input()});
    auto reshape1_m = wrap_type<ov::op::v1::Reshape>({transpose_inner_m, any_input()});
    auto reshape2_m = optional<ov::op::v1::Reshape>({reshape1_m, any_input()});
    auto transpose_outer_m = wrap_type<ov::op::v1::Transpose>({reshape2_m, any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto transpose_outer = ov::as_type_ptr<ov::op::v1::Transpose>(m.get_match_root());
        if (!transpose_outer || !is_transpose_0213(transpose_outer))
            return false;

        auto transpose_inner = ov::as_type_ptr<ov::op::v1::Transpose>(
            pm.at(transpose_inner_m).get_node_shared_ptr());
        if (!transpose_inner || !is_transpose_0213(transpose_inner))
            return false;

        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(
            pm.at(rope_m).get_node_shared_ptr());
        if (!rope || !rope->get_config().input_trans0213)
            return false;

        // RoPE with input_trans0213 outputs [B,H,L,D].
        // Transpose(0213) converts to [B,L,H,D], Reshapes are identity, outer Transpose(0213)
        // converts back to [B,H,L,D]. Net effect is identity — connect RoPE directly to consumers.
        auto rope_output = rope->output(0);
        transpose_outer->output(0).replace(rope_output);

        GPU_DEBUG_LOG << "[WebNNRoPETransposeElim] Eliminated Transpose-Reshape-Transpose chain for "
                      << rope->get_friendly_name() << std::endl;

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(transpose_outer_m, "WebNNRoPETransposeEliminationKV");
    this->register_matcher(matcher, callback);
}

WebNNRoPETransposeElimination::WebNNRoPETransposeElimination() {
    add_matcher<WebNNRoPETransposeEliminationKV>();
}

}  // namespace ov::intel_gpu

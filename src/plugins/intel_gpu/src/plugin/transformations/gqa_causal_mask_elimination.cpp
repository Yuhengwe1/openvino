// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_causal_mask_elimination.hpp"

#include <memory>

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

namespace {

// Trace back through the mask input to verify it comes from the GQA causal mask pattern:
//   Squeeze ← Select ← [Clamp] ← Less ← CumSum(exclusive=true)
// Clamp is optional: newer ORT GQA decompositions emit Select(Less(...), 0, -inf) directly.
bool is_gqa_causal_mask_pattern(const ov::Output<ov::Node>& mask_output) {
    auto node = mask_output.get_node_shared_ptr();

    auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(node);
    if (!squeeze)
        return false;

    auto select = ov::as_type_ptr<ov::op::v1::Select>(squeeze->input_value(0).get_node_shared_ptr());
    if (!select)
        return false;

    auto condition_node = select->input_value(0).get_node_shared_ptr();

    std::shared_ptr<ov::op::v1::Less> less;
    auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(condition_node);
    if (clamp) {
        less = ov::as_type_ptr<ov::op::v1::Less>(clamp->input_value(0).get_node_shared_ptr());
    } else {
        less = ov::as_type_ptr<ov::op::v1::Less>(condition_node);
    }

    if (!less)
        return false;

    // CumSum may be on either input of Less (column or row range)
    for (size_t i = 0; i < 2; ++i) {
        auto cumsum = ov::as_type_ptr<ov::op::v0::CumSum>(less->input_value(i).get_node_shared_ptr());
        if (cumsum && cumsum->is_exclusive())
            return true;
    }

    return false;
}

}  // namespace

WebNNGQACausalMaskEliminationSDPA::WebNNGQACausalMaskEliminationSDPA() {
    using namespace ov::pass::pattern;

    auto sdpa_m = wrap_type<ov::intel_gpu::op::IndirectSDPA>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto indirect_sdpa = ov::as_type_ptr<ov::intel_gpu::op::IndirectSDPA>(m.get_match_root());
        if (!indirect_sdpa)
            return false;

        auto input_count = indirect_sdpa->get_input_size();

        if (input_count < 5)
            return false;

        if (indirect_sdpa->get_causal())
            return false;

        auto mask_input = indirect_sdpa->input_value(3);

        if (!is_gqa_causal_mask_pattern(mask_input))
            return false;

        OutputVector new_data_inputs;
        new_data_inputs.push_back(indirect_sdpa->input_value(0));  // Q
        new_data_inputs.push_back(indirect_sdpa->input_value(1));  // K
        new_data_inputs.push_back(indirect_sdpa->input_value(2));  // V
        for (size_t i = 4; i < input_count - 1; ++i) {
            new_data_inputs.push_back(indirect_sdpa->input_value(i));
        }

        auto beam_table = indirect_sdpa->input_value(input_count - 1);

        auto new_sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(
            new_data_inputs,
            beam_table,
            true,  // is_causal = true
            indirect_sdpa->get_indirect_axis(),
            indirect_sdpa->get_input0_transpose_order(),
            indirect_sdpa->get_input1_transpose_order(),
            indirect_sdpa->get_input2_transpose_order(),
            indirect_sdpa->get_output_transpose_order(),
            indirect_sdpa->get_output_type());

        new_sdpa->set_friendly_name(indirect_sdpa->get_friendly_name());
        ov::copy_runtime_info(indirect_sdpa, new_sdpa);
        ov::replace_node(indirect_sdpa, new_sdpa);

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "WebNNGQACausalMaskEliminationSDPA");
    this->register_matcher(matcher, callback);
}

WebNNGQACausalMaskElimination::WebNNGQACausalMaskElimination() {
    add_matcher<WebNNGQACausalMaskEliminationSDPA>();
}

}  // namespace ov::intel_gpu

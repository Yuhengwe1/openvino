// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_fusion.hpp"

#include <memory>
#include <vector>

#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

bool is_sign_flip_constant(const std::shared_ptr<ov::op::v0::Constant>& constant) {
    auto shape = constant->get_shape();
    size_t total = ov::shape_size(shape);
    if (total != 2)
        return false;

    auto values = constant->cast_vector<float>();
    return (values[0] == -1.0f && values[1] == 1.0f) || (values[0] == 1.0f && values[1] == -1.0f);
}

size_t get_half_head_dim(const ov::PartialShape& shape_5d) {
    if (shape_5d.rank().get_length() != 5)
        return 0;
    if (shape_5d[3].is_dynamic() || shape_5d[3].get_length() != 2)
        return 0;
    if (shape_5d[4].is_dynamic())
        return 0;
    return shape_5d[4].get_length();
}

// Get the split source node from a Concat that swaps halves.
// Supports both v1::VariadicSplit and v1::Split.
// Returns the split input value and validates axis/halves-swapped.
struct SplitInfo {
    ov::Output<ov::Node> split_input;
    bool halves_swapped = false;
    std::shared_ptr<ov::Node> split_node;
};

bool get_split_from_concat(const std::shared_ptr<ov::op::v0::Concat>& concat_node,
                           int64_t expected_axis,
                           SplitInfo& info) {
    if (concat_node->get_input_size() != 2)
        return false;

    auto concat_in0 = concat_node->input_value(0);
    auto concat_in1 = concat_node->input_value(1);

    auto concat_in0_node = concat_in0.get_node_shared_ptr();
    auto concat_in1_node = concat_in1.get_node_shared_ptr();

    // Check VariadicSplit
    auto try_variadic_split = [&]() -> bool {
        auto vs0 = ov::as_type_ptr<ov::op::v1::VariadicSplit>(concat_in0_node);
        auto vs1 = ov::as_type_ptr<ov::op::v1::VariadicSplit>(concat_in1_node);

        if (!vs0 || !vs1 || vs0.get() != vs1.get())
            return false;

        auto split_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(vs0->input_value(1).get_node_shared_ptr());
        if (!split_axis_node)
            return false;

        auto split_axis_val = split_axis_node->cast_vector<int64_t>();
        if (split_axis_val.size() != 1)
            return false;

        int64_t split_axis = split_axis_val[0];
        auto input_rank = vs0->get_input_partial_shape(0).rank();
        if (input_rank.is_dynamic())
            return false;
        if (split_axis < 0)
            split_axis += input_rank.get_length();
        if (split_axis != expected_axis)
            return false;

        size_t port0 = concat_in0.get_index();
        size_t port1 = concat_in1.get_index();
        info.halves_swapped = (port0 == 1 && port1 == 0);
        if (!info.halves_swapped && !(port0 == 0 && port1 == 1))
            return false;

        info.split_input = vs0->input_value(0);
        info.split_node = vs0;
        return true;
    };

    // Check v1::Split
    auto try_split = [&]() -> bool {
        auto s0 = ov::as_type_ptr<ov::op::v1::Split>(concat_in0_node);
        auto s1 = ov::as_type_ptr<ov::op::v1::Split>(concat_in1_node);

        if (!s0 || !s1 || s0.get() != s1.get())
            return false;

        if (s0->get_num_splits() != 2)
            return false;

        auto split_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(s0->input_value(1).get_node_shared_ptr());
        if (!split_axis_node)
            return false;

        auto split_axis_val = split_axis_node->cast_vector<int64_t>();
        if (split_axis_val.size() != 1)
            return false;

        int64_t split_axis = split_axis_val[0];
        auto input_rank = s0->get_input_partial_shape(0).rank();
        if (input_rank.is_dynamic())
            return false;
        if (split_axis < 0)
            split_axis += input_rank.get_length();
        if (split_axis != expected_axis)
            return false;

        size_t port0 = concat_in0.get_index();
        size_t port1 = concat_in1.get_index();
        info.halves_swapped = (port0 == 1 && port1 == 0);
        if (!info.halves_swapped && !(port0 == 0 && port1 == 1))
            return false;

        info.split_input = s0->input_value(0);
        info.split_node = s0;
        return true;
    };

    return try_variadic_split() || try_split();
}

}  // namespace

WebNNRoPEFusionMatcher::WebNNRoPEFusionMatcher() {
    using namespace ov::pass::pattern;

    auto add_pattern = wrap_type<ov::op::v1::Add>({any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto add_node = ov::as_type_ptr<ov::op::v1::Add>(m.get_match_root());
        if (!add_node)
            return false;

        auto add_output_shape = add_node->get_output_partial_shape(0);
        if (add_output_shape.rank().is_dynamic())
            return false;

        auto add_rank = add_output_shape.rank().get_length();
        if (add_rank != 4 && add_rank != 5)
            return false;

        auto input0 = add_node->input_value(0).get_node_shared_ptr();
        auto input1 = add_node->input_value(1).get_node_shared_ptr();

        // Case 1 (4D Add): Add(Reshape(mul_cos_5d, 4d), Reshape(mul_sin_5d, 4d))
        // Case 2 (5D Add): Add(mul_cos_5d, mul_sin_5d) with output reshape to 4D
        std::shared_ptr<ov::op::v1::Multiply> mul0 = nullptr;
        std::shared_ptr<ov::op::v1::Multiply> mul1 = nullptr;
        std::shared_ptr<ov::op::v1::Reshape> reshape_before_add_0 = nullptr;
        std::shared_ptr<ov::op::v1::Reshape> reshape_before_add_1 = nullptr;

        if (add_rank == 4) {
            auto reshape0 = ov::as_type_ptr<ov::op::v1::Reshape>(input0);
            auto reshape1 = ov::as_type_ptr<ov::op::v1::Reshape>(input1);
            if (!reshape0 || !reshape1)
                return false;

            auto get_5d_multiply = [](const std::shared_ptr<ov::op::v1::Reshape>& reshape)
                -> std::shared_ptr<ov::op::v1::Multiply> {
                auto input_shape = reshape->get_input_partial_shape(0);
                if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 5)
                    return nullptr;
                return ov::as_type_ptr<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
            };

            mul0 = get_5d_multiply(reshape0);
            mul1 = get_5d_multiply(reshape1);
            if (!mul0 || !mul1)
                return false;

            reshape_before_add_0 = reshape0;
            reshape_before_add_1 = reshape1;
        } else {
            // 5D Add: inputs are directly Multiplies
            mul0 = ov::as_type_ptr<ov::op::v1::Multiply>(input0);
            mul1 = ov::as_type_ptr<ov::op::v1::Multiply>(input1);
            if (!mul0 || !mul1)
                return false;

            auto shape0 = mul0->get_output_partial_shape(0);
            if (shape0.rank().is_dynamic() || shape0.rank().get_length() != 5)
                return false;
        }

        // Identify which is mul_cos and which is mul_sin.
        // mul_sin branch has a Concat in its ancestry (possibly through a sign multiply).
        // WebNN structure: mul_sin_signed = Multiply(Multiply(Concat, sin), sign_const)
        //            or:   mul_sin_signed = Multiply(sign_const, Multiply(Concat, sin))
        // mul_cos structure: Multiply(x_5d, cos_5d)

        // Find Concat in a multiply's input tree (up to 2 levels deep)
        auto find_concat_in_mul = [](const std::shared_ptr<ov::op::v1::Multiply>& mul)
            -> std::shared_ptr<ov::op::v0::Concat> {
            for (size_t i = 0; i < 2; i++) {
                auto inp = mul->input_value(i).get_node_shared_ptr();
                if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(inp))
                    return concat;
                if (auto inner_mul = ov::as_type_ptr<ov::op::v1::Multiply>(inp)) {
                    for (size_t j = 0; j < 2; j++) {
                        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(
                                inner_mul->input_value(j).get_node_shared_ptr()))
                            return concat;
                    }
                }
            }
            return nullptr;
        };

        auto concat_from_mul0 = find_concat_in_mul(mul0);
        auto concat_from_mul1 = find_concat_in_mul(mul1);

        std::shared_ptr<ov::op::v1::Multiply> mul_cos = nullptr;
        std::shared_ptr<ov::op::v1::Multiply> mul_sin_outer = nullptr;
        std::shared_ptr<ov::op::v0::Concat> concat_node = nullptr;
        std::shared_ptr<ov::op::v1::Reshape> reshape_cos = nullptr;
        std::shared_ptr<ov::op::v1::Reshape> reshape_sin = nullptr;

        if (concat_from_mul1 && !concat_from_mul0) {
            mul_cos = mul0;
            mul_sin_outer = mul1;
            concat_node = concat_from_mul1;
            reshape_cos = reshape_before_add_0;
            reshape_sin = reshape_before_add_1;
        } else if (concat_from_mul0 && !concat_from_mul1) {
            mul_cos = mul1;
            mul_sin_outer = mul0;
            concat_node = concat_from_mul0;
            reshape_cos = reshape_before_add_1;
            reshape_sin = reshape_before_add_0;
        } else {
            return false;
        }

        // Verify concat axis is 3 (the "2" dimension in [B,L,H,2,half_D])
        auto concat_output_shape = concat_node->get_output_partial_shape(0);
        if (concat_output_shape.rank().is_dynamic())
            return false;

        auto concat_rank = concat_output_shape.rank().get_length();
        int64_t concat_axis = concat_node->get_axis();
        if (concat_axis < 0)
            concat_axis += concat_rank;
        if (concat_rank != 5 || concat_axis != 3)
            return false;

        // Verify Concat comes from a Split with halves swapped
        SplitInfo split_info;
        if (!get_split_from_concat(concat_node, 3, split_info))
            return false;

        auto split_input = split_info.split_input;

        // Now extract the structure from mul_sin_outer.
        // WebNN: mul_sin_outer = Multiply(Multiply(Concat, sin), sign_const)
        //   or:  mul_sin_outer = Multiply(sign_const, Multiply(Concat, sin))
        // The sign constant has shape with total 2 elements, values [-1, 1].
        // The inner Multiply is Multiply(Concat, sin).

        std::shared_ptr<ov::op::v0::Constant> sign_constant = nullptr;
        std::shared_ptr<ov::op::v1::Multiply> mul_sin_inner = nullptr;
        ov::Output<ov::Node> sin_value;

        for (size_t i = 0; i < 2; i++) {
            auto outer_input = mul_sin_outer->input_value(i).get_node_shared_ptr();
            auto other_outer = mul_sin_outer->input_value(1 - i).get_node_shared_ptr();

            // Check if this input is the sign constant
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(outer_input)) {
                if (is_sign_flip_constant(constant)) {
                    sign_constant = constant;
                    // The other input should be the inner Multiply(Concat, sin)
                    if (auto inner = ov::as_type_ptr<ov::op::v1::Multiply>(other_outer)) {
                        mul_sin_inner = inner;
                    }
                    break;
                }
            }
            // Check if the other input is the sign constant
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(other_outer)) {
                if (is_sign_flip_constant(constant)) {
                    sign_constant = constant;
                    // This input should be the inner Multiply(Concat, sin)
                    if (auto inner = ov::as_type_ptr<ov::op::v1::Multiply>(outer_input)) {
                        mul_sin_inner = inner;
                    }
                    break;
                }
            }
        }

        // If no sign constant found, mul_sin_outer might directly be Multiply(Concat, sin)
        if (!sign_constant) {
            mul_sin_inner = mul_sin_outer;
        }

        if (!mul_sin_inner)
            return false;

        // Extract sin_value from mul_sin_inner = Multiply(Concat, sin) or Multiply(sin, Concat)
        bool found_sin = false;
        for (size_t i = 0; i < 2; i++) {
            auto inp = mul_sin_inner->input_value(i).get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::Concat>(inp)) {
                sin_value = mul_sin_inner->input_value(1 - i);
                found_sin = true;
                break;
            }
        }
        if (!found_sin)
            return false;

        // Verify mul_cos uses the same 5D tensor as the split input
        // mul_cos = Multiply(x_5d, cos_5d) where x_5d == split_input
        ov::Output<ov::Node> cos_value;
        ov::Output<ov::Node> x_5d;
        bool found_x_in_mul_cos = false;

        for (size_t i = 0; i < 2; i++) {
            if (mul_cos->input_value(i) == split_input) {
                x_5d = split_input;
                cos_value = mul_cos->input_value(1 - i);
                found_x_in_mul_cos = true;
                break;
            }
        }

        if (!found_x_in_mul_cos)
            return false;

        // Verify x_5d is actually 5D with shape [B,L,H,2,half_D]
        auto x_5d_shape = x_5d.get_partial_shape();
        if (x_5d_shape.rank().is_dynamic() || x_5d_shape.rank().get_length() != 5)
            return false;

        size_t half_head_dim = get_half_head_dim(x_5d_shape);
        if (half_head_dim == 0)
            return false;

        size_t rotary_ndims = 2 * half_head_dim;

        // Trace x_5d back through reshape to get original 3D/4D input
        auto x_5d_node = x_5d.get_node_shared_ptr();
        std::shared_ptr<ov::op::v1::Reshape> input_reshape = nullptr;
        ov::Output<ov::Node> rope_data_input;

        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(x_5d_node)) {
            auto reshape_input_shape = reshape->get_input_partial_shape(0);
            if (reshape_input_shape.rank().is_dynamic())
                return false;

            auto input_rank = reshape_input_shape.rank().get_length();
            if (input_rank == 3 || input_rank == 4) {
                input_reshape = reshape;
                rope_data_input = reshape->input_value(0);
            }
        }

        if (!input_reshape)
            return false;

        // Trace cos/sin through reshapes to get their original values (Gather output)
        auto trace_through_reshape = [](ov::Output<ov::Node> val) -> ov::Output<ov::Node> {
            auto node = val.get_node_shared_ptr();
            while (ov::is_type<ov::op::v1::Reshape>(node)) {
                auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
                val = reshape->input_value(0);
                node = val.get_node_shared_ptr();
            }
            return val;
        };

        auto cos_original = trace_through_reshape(cos_value);
        auto sin_original = trace_through_reshape(sin_value);

        // The RoPE kernel's RotateHalf path indexes cos/sin as:
        //   4D: INPUT1_GET_INDEX(batch, head_clamped, position, 0) + r
        //   3D: INPUT1_GET_INDEX(batch, head_as_feature!, 0, 0) -- WRONG for our case
        // cos/sin from Gather are [B, L, half_D] (3D). The kernel's 3D path would
        // incorrectly use head index to address the position dimension.
        // Solution: reshape cos/sin to 4D [B, 1, L, half_D] so the kernel uses the
        // 4D path with head=1 (broadcast) and position correctly addressed.
        auto ensure_4d_cos_sin = [](ov::Output<ov::Node> val) -> ov::Output<ov::Node> {
            auto shape = val.get_partial_shape();
            if (shape.rank().is_static() && shape.rank().get_length() == 3) {
                // [B, L, half_D] -> [B, 1, L, half_D]
                auto target = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 0, 0});
                // Use Unsqueeze at axis=1: [B, L, half_D] -> [B, 1, L, half_D]
                auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
                auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(val, axis);
                return unsqueeze->output(0);
            }
            return val;
        };

        auto cos_4d = ensure_4d_cos_sin(cos_original);
        auto sin_4d = ensure_4d_cos_sin(sin_original);

        // Build replacement RoPE node
        ov::op::internal::RoPE::Config config;
        config.rotary_ndims = rotary_ndims;
        config.head_size = rotary_ndims;
        config.cos_sin_ndims = half_head_dim;  // ensures COS_SIN_TABLE_OFFSET=0
        config.input_trans0213 = true;

        auto input_data_shape = rope_data_input.get_partial_shape();
        auto input_data_rank = input_data_shape.rank().get_length();

        size_t head_count = 0;
        if (x_5d_shape[2].is_static()) {
            head_count = x_5d_shape[2].get_length();
        }

        ov::Output<ov::Node> rope_input_data;

        if (input_data_rank == 4) {
            rope_input_data = rope_data_input;
        } else if (input_data_rank == 3 && head_count > 0) {
            auto target_shape = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{0, 0, static_cast<int64_t>(head_count), static_cast<int64_t>(rotary_ndims)});
            auto new_reshape = std::make_shared<ov::op::v1::Reshape>(rope_data_input, target_shape, true);
            rope_input_data = new_reshape->output(0);
        } else {
            return false;
        }

        ov::OutputVector new_args;
        new_args.push_back(rope_input_data);
        new_args.push_back(cos_4d);
        new_args.push_back(sin_4d);

        auto new_rope = std::make_shared<ov::op::internal::RoPE>(new_args, config);

        // RoPE with input_trans0213 outputs [B, H, L, D]; transpose back to [B, L, H, D]
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto output_transpose = std::make_shared<ov::op::v1::Transpose>(new_rope->output(0), transpose_order);

        // Determine which node to replace in the graph
        std::shared_ptr<ov::Node> node_to_replace;
        if (add_rank == 4) {
            // 4D case: replace the Add node itself (reshapes are inputs, not outputs)
            node_to_replace = add_node;
        } else {
            // 5D case: find the output reshape (5D→4D) after the Add
            std::shared_ptr<ov::op::v1::Reshape> output_reshape = nullptr;
            auto add_consumers = add_node->output(0).get_target_inputs();
            for (auto& consumer : add_consumers) {
                if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(consumer.get_node()->shared_from_this())) {
                    auto out_shape = reshape->get_output_partial_shape(0);
                    if (out_shape.rank().is_static() && out_shape.rank().get_length() == 4) {
                        output_reshape = reshape;
                        break;
                    }
                }
            }
            if (!output_reshape)
                return false;
            node_to_replace = output_reshape;
        }

        output_transpose->set_friendly_name(node_to_replace->get_friendly_name());

        ov::NodeVector replaced_nodes;
        replaced_nodes.push_back(input_reshape);
        replaced_nodes.push_back(split_info.split_node);
        replaced_nodes.push_back(concat_node);
        replaced_nodes.push_back(mul_cos);
        replaced_nodes.push_back(mul_sin_inner);
        if (sign_constant) {
            replaced_nodes.push_back(mul_sin_outer);
        }
        if (reshape_cos)
            replaced_nodes.push_back(reshape_cos);
        if (reshape_sin)
            replaced_nodes.push_back(reshape_sin);
        replaced_nodes.push_back(add_node);
        if (node_to_replace != add_node)
            replaced_nodes.push_back(node_to_replace);

        ov::copy_runtime_info(replaced_nodes, {new_rope, output_transpose});
        ov::replace_node(node_to_replace, output_transpose);

        register_new_node(new_rope);

        GPU_DEBUG_LOG << "[WebNNRoPEFusion] Fused RoPE: " << new_rope->get_friendly_name()
                      << " rotary_ndims=" << config.rotary_ndims
                      << " heads=" << head_count
                      << std::endl;

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_pattern, "WebNNRoPEFusionMatcher");
    this->register_matcher(m, callback);
}

WebNNRoPEFusion::WebNNRoPEFusion() {
    add_matcher<WebNNRoPEFusionMatcher>();
}

// ============================================================================
// WebNNRoPEGatherFusionMatcher
//
// After WebNNRoPEFusion creates RoPE(data, gathered_cos, gathered_sin),
// this pass detects that cos/sin inputs come from Gather(table, position_ids, axis)
// and folds the gather into the RoPE node:
//   RoPE(data, cos_table, sin_table, position_ids) with gather_position_arg_id=3
//
// This eliminates 2 separate Gather GPU kernel dispatches per RoPE invocation
// (44 Gathers total for Q+K across 22 layers).
// ============================================================================

WebNNRoPEGatherFusionMatcher::WebNNRoPEGatherFusionMatcher() {
    using namespace ov::pass::pattern;

    auto rope_m = wrap_type<ov::op::internal::RoPE>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto rope_node = ov::as_type_ptr<ov::op::internal::RoPE>(m.get_match_root());
        if (!rope_node)
            return false;

        auto config = rope_node->get_config();

        // Skip if gather is already fused
        if (config.gather_position_arg_id > 0)
            return false;

        // RoPE should have exactly 3 inputs: [data, cos, sin]
        if (rope_node->get_input_size() != 3)
            return false;

        auto cos_input = rope_node->input_value(1);
        auto sin_input = rope_node->input_value(2);

        // Trace through Unsqueeze if present (added by ensure_4d_cos_sin in RoPE fusion)
        auto trace_through_unsqueeze = [](ov::Output<ov::Node> val)
            -> std::pair<ov::Output<ov::Node>, std::shared_ptr<ov::Node>> {
            auto node = val.get_node_shared_ptr();
            if (auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(node)) {
                return {unsqueeze->input_value(0), unsqueeze};
            }
            return {val, nullptr};
        };

        auto [cos_pre_unsqueeze, cos_unsqueeze] = trace_through_unsqueeze(cos_input);
        auto [sin_pre_unsqueeze, sin_unsqueeze] = trace_through_unsqueeze(sin_input);

        // Check if cos/sin come from Gather ops
        auto cos_gather = ov::as_type_ptr<ov::op::v8::Gather>(cos_pre_unsqueeze.get_node_shared_ptr());
        auto sin_gather = ov::as_type_ptr<ov::op::v8::Gather>(sin_pre_unsqueeze.get_node_shared_ptr());

        if (!cos_gather || !sin_gather)
            return false;

        // Both gathers must use the same position_ids (input 1) and axis (input 2)
        auto cos_position_ids = cos_gather->input_value(1);
        auto sin_position_ids = sin_gather->input_value(1);

        if (cos_position_ids != sin_position_ids)
            return false;

        // Verify gather axis is a constant and same for both
        auto cos_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(
            cos_gather->input_value(2).get_node_shared_ptr());
        auto sin_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(
            sin_gather->input_value(2).get_node_shared_ptr());

        if (!cos_axis_const || !sin_axis_const)
            return false;

        auto cos_axis_val = cos_axis_const->cast_vector<int64_t>();
        auto sin_axis_val = sin_axis_const->cast_vector<int64_t>();
        if (cos_axis_val.size() != 1 || sin_axis_val.size() != 1)
            return false;
        if (cos_axis_val[0] != sin_axis_val[0])
            return false;

        // Get the cos/sin tables (input 0 of Gather)
        auto cos_table = cos_gather->input_value(0);
        auto sin_table = sin_gather->input_value(0);

        // Verify cos/sin tables are the same rank (should be 4D: [1, 1, max_seq_len, half_D])
        auto cos_table_shape = cos_table.get_partial_shape();
        auto sin_table_shape = sin_table.get_partial_shape();
        if (cos_table_shape.rank().is_dynamic() || sin_table_shape.rank().is_dynamic())
            return false;

        auto cos_table_rank = cos_table_shape.rank().get_length();
        auto sin_table_rank = sin_table_shape.rank().get_length();
        if (cos_table_rank != sin_table_rank)
            return false;

        // Tables must be 2D or 4D for the RoPE kernel to handle correctly
        if (cos_table_rank != 2 && cos_table_rank != 4)
            return false;

        // Get position_ids and determine its rank for the kernel's GATHER_RANK
        ov::Output<ov::Node> position_ids = cos_position_ids;
        auto position_ids_shape = position_ids.get_partial_shape();
        if (position_ids_shape.rank().is_dynamic())
            return false;

        auto position_ids_rank = position_ids_shape.rank().get_length();

        // Convert position_ids to i32 if needed (kernel reads as int/uint)
        auto pos_elem_type = position_ids.get_element_type();
        if (pos_elem_type != ov::element::i32) {
            auto convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
            position_ids = convert->output(0);
        }

        // The RoPE kernel's ENABLE_GATHER path for non-4D expects position_ids to be at
        // least 2D [batch, seq_len] so that INPUT3_FEATURE_NUM = seq_len and the kernel
        // indexes gather[batch, position] correctly. If position_ids is 1D [seq_len],
        // reshape it to [1, seq_len] to match the expected layout.
        if (position_ids_rank == 1) {
            auto target_shape = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, -1});
            auto reshape = std::make_shared<ov::op::v1::Reshape>(position_ids, target_shape, false);
            position_ids = reshape->output(0);
        }

        // Build the new RoPE node with 4 inputs: [data, cos_table, sin_table, position_ids]
        ov::OutputVector new_args;
        new_args.push_back(rope_node->input_value(0));  // data (unchanged)
        new_args.push_back(cos_table);                   // cos table (full, ungathered)
        new_args.push_back(sin_table);                   // sin table (full, ungathered)
        new_args.push_back(position_ids);                // position_ids for gather

        config.gather_position_arg_id = 3;

        auto new_rope = std::make_shared<ov::op::internal::RoPE>(new_args, config);
        new_rope->set_friendly_name(rope_node->get_friendly_name());

        ov::NodeVector replaced_nodes;
        replaced_nodes.push_back(rope_node);
        if (cos_unsqueeze)
            replaced_nodes.push_back(cos_unsqueeze);
        if (sin_unsqueeze)
            replaced_nodes.push_back(sin_unsqueeze);
        replaced_nodes.push_back(cos_gather);
        replaced_nodes.push_back(sin_gather);

        ov::copy_runtime_info(replaced_nodes, new_rope);
        ov::replace_node(rope_node, new_rope);

        GPU_DEBUG_LOG << "[WebNNRoPEGatherFusion] Folded gather into RoPE: "
                      << new_rope->get_friendly_name()
                      << " position_ids_rank=" << position_ids_shape.rank().get_length()
                      << " cos_table_rank=" << cos_table_rank
                      << std::endl;

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(rope_m, "WebNNRoPEGatherFusionMatcher");
    this->register_matcher(matcher, callback);
}

WebNNRoPEGatherFusion::WebNNRoPEGatherFusion() {
    add_matcher<WebNNRoPEGatherFusionMatcher>();
}

}  // namespace ov::intel_gpu

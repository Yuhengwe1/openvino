#include "kv_cache_expand_transform.hpp"
#include <memory>

#include "intel_gpu/op/kv_cache.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {
KVCacheExpandMatcher::KVCacheExpandMatcher() {
    using namespace ov::pass::pattern;

    auto kv_cache = wrap_type<op::KVCache>();
    auto reshape = wrap_type<ov::op::v1::Reshape>({kv_cache, wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] transformation callback returns true, skip expansion" << std::endl;
            return false;
        }

        auto reshape_node = ov::as_type_ptr<ov::op::v1::Reshape>(m.get_match_root());
        if (!reshape_node) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] failed to match reshape pattern, skip expansion" << std::endl;
             return false;
        }

        auto kv_cache_node = ov::as_type_ptr<op::KVCache>(reshape_node->get_input_node_shared_ptr(0));

        if (!kv_cache_node) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] failed to match reshape pattern with KVCache input, skip expansion" << std::endl;
            return false;
        }

        // Check if reshape inserts one dim only
        auto input_shape = reshape_node->get_input_partial_shape(0);
        auto output_shape = reshape_node->get_output_partial_shape(0);

        if (input_shape.rank().get_length() >= output_shape.rank().get_length()) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] found unsupported reshape pattern which is not inserting one dim, skip expansion" << std::endl;
            return false;
        }

        std::vector<int64_t> axes;
        size_t input_idx = 0;
        auto input_shape_rank = input_shape.rank().get_length();
        auto output_shape_rank = output_shape.rank().get_length();
        for (size_t i = 0; i < output_shape_rank; ++i) {
            if (input_idx < input_shape_rank && input_shape[input_idx].same_scheme(output_shape[i])) {
                input_idx++;
            } else if (output_shape[i].is_static() && output_shape[i].get_length() == 1) {
                axes.push_back(i);
            } else {
                GPU_DEBUG_LOG << "[KVCacheExpandMatcher] found unsupported reshape pattern which is not inserting one dim, skip expansion" << std::endl;
                return false;
            }
        }

        if (input_idx != input_shape_rank) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] found unsupported reshape pattern which is not inserting one dim, skip expansion" << std::endl;
            return false;
        }

        bool expansion_pattern = false;
        for (auto& consumer : reshape_node->output(0).get_target_inputs()) {
            auto consumer_node = consumer.get_node();
            if (ov::is_type<ov::op::v3::Broadcast>(consumer_node) || ov::is_type<ov::op::v1::Broadcast>(consumer_node)) {
                 for (auto& b_consumer : consumer_node->output(0).get_target_inputs()) {
                     if (ov::is_type<ov::op::v1::Reshape>(b_consumer.get_node())) {
                         expansion_pattern = true;
                         break;
                     }
                 }
            }
            if (expansion_pattern) break;
        }

        if (!expansion_pattern) {
            GPU_DEBUG_LOG << "[KVCacheExpandMatcher] failed to find kv_cache expansion pattern, skip expansion" << std::endl;
            return false;
        }

        if (axes.empty())
            return false;

        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(reshape_node->input_value(0), axes_node);
        unsqueeze->set_friendly_name(reshape_node->get_friendly_name());
        ov::copy_runtime_info(reshape_node, unsqueeze);
        ov::replace_node(reshape_node, unsqueeze);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape, "KVCacheExpansionMatcher");
    this->register_matcher(m, callback);
}

ov::intel_gpu::KVCacheExpandTransformation::KVCacheExpandTransformation() {
    add_matcher<KVCacheExpandMatcher>();
}
}

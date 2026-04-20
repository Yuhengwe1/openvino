#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class KVCacheExpandMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheExpandMatcher");
    KVCacheExpandMatcher();
};

class KVCacheExpandTransformation : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("KVCacheExpandTransformation");
    KVCacheExpandTransformation();
};

}   // namespace ov::intel_gpu
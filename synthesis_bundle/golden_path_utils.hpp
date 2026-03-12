#pragma once

#include <filesystem>
#include <string>

namespace eagle4 {
namespace test_paths {

struct GoldenPaths {
    std::string tensor_dir;
    std::string packed_dir;
    std::string norm_dir;
    std::string lm_dir;
};

inline std::string ensure_trailing_slash(const std::filesystem::path& path) {
    std::string out = path.string();
    if (!out.empty() && out.back() != '/') out.push_back('/');
    return out;
}

inline std::filesystem::path find_workspace_root(
    std::filesystem::path start = std::filesystem::current_path()) {
    std::error_code ec;
    for (;;) {
        if (std::filesystem::exists(start / "hardware", ec) &&
            std::filesystem::exists(start / "sglang-eagle4", ec)) {
            return start;
        }
        if (start.empty() || start == start.root_path()) break;
        start = start.parent_path();
    }
    return {};
}

inline GoldenPaths default_eagle4_golden_paths(
    std::filesystem::path start = std::filesystem::current_path()) {
    const auto root = find_workspace_root(std::move(start));
    if (!root.empty()) {
        return {
            ensure_trailing_slash(
                root / "capture/eagle_verified_pipeline_4bit/cpmcu_tensors"),
            ensure_trailing_slash(root / "hardware/EAGLE/eagle/hls_hw/packed_all"),
            ensure_trailing_slash(
                root / "capture/eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit"),
            ensure_trailing_slash(
                root / "capture/eagle_verified_pipeline_4bit/hls_4bit/lm_head"),
        };
    }
    return {
        "../eagle_verified_pipeline_4bit/cpmcu_tensors/",
        "../packed_all/",
        "../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/",
        "../eagle_verified_pipeline_4bit/hls_4bit/lm_head/",
    };
}

}  // namespace test_paths
}  // namespace eagle4

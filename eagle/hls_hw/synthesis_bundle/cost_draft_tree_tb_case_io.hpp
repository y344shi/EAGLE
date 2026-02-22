#ifndef TMAC_COST_DRAFT_TREE_TB_CASE_IO_HPP
#define TMAC_COST_DRAFT_TREE_TB_CASE_IO_HPP

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace tmac {
namespace hls {
namespace tb_case_io {

using RawCaseMap = std::unordered_map<std::string, std::vector<std::string>>;

inline bool parse_key_count_file(const std::string& path,
                                 RawCaseMap* out,
                                 std::string* err_msg) {
    std::ifstream in(path);
    if (!in) {
        if (err_msg != nullptr) {
            *err_msg = "cannot open input file: " + path;
        }
        return false;
    }

    out->clear();

    std::string key;
    while (in >> key) {
        if (!key.empty() && key[0] == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        size_t count = 0;
        if (!(in >> count)) {
            if (err_msg != nullptr) {
                *err_msg = "failed reading count for key: " + key;
            }
            return false;
        }

        std::vector<std::string> payload(count);
        for (size_t i = 0; i < count; ++i) {
            if (!(in >> payload[i])) {
                if (err_msg != nullptr) {
                    *err_msg = "failed reading payload token for key: " + key;
                }
                return false;
            }
        }

        (*out)[key] = std::move(payload);
    }

    return true;
}

inline bool has_key(const RawCaseMap& kv, const std::string& key) {
    return kv.find(key) != kv.end();
}

inline bool parse_int_token(const std::string& s, int* out) {
    try {
        const long long v = std::stoll(s);
        if (v < static_cast<long long>(std::numeric_limits<int>::min()) ||
            v > static_cast<long long>(std::numeric_limits<int>::max())) {
            return false;
        }
        *out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parse_i32_token(const std::string& s, int32_t* out) {
    try {
        const long long v = std::stoll(s);
        if (v < static_cast<long long>(std::numeric_limits<int32_t>::min()) ||
            v > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        *out = static_cast<int32_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parse_i64_token(const std::string& s, int64_t* out) {
    try {
        *out = static_cast<int64_t>(std::stoll(s));
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parse_float_token(const std::string& s, float* out) {
    try {
        *out = std::stof(s);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool read_int_array(const RawCaseMap& kv,
                           const std::string& key,
                           size_t expected_count,
                           std::vector<int>* out,
                           std::string* err_msg,
                           bool required = true) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        if (required && err_msg != nullptr) {
            *err_msg = "missing key: " + key;
        }
        return !required;
    }

    const std::vector<std::string>& payload = it->second;
    if (expected_count != static_cast<size_t>(-1) && payload.size() != expected_count) {
        if (err_msg != nullptr) {
            *err_msg = "size mismatch for " + key + ": got=" +
                       std::to_string(payload.size()) + " expected=" +
                       std::to_string(expected_count);
        }
        return false;
    }

    out->assign(payload.size(), 0);
    for (size_t i = 0; i < payload.size(); ++i) {
        if (!parse_int_token(payload[i], &((*out)[i]))) {
            if (err_msg != nullptr) {
                *err_msg = "invalid int payload for " + key + " at index " +
                           std::to_string(i);
            }
            return false;
        }
    }

    return true;
}

inline bool read_i32_array(const RawCaseMap& kv,
                           const std::string& key,
                           size_t expected_count,
                           std::vector<int32_t>* out,
                           std::string* err_msg,
                           bool required = true) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        if (required && err_msg != nullptr) {
            *err_msg = "missing key: " + key;
        }
        return !required;
    }

    const std::vector<std::string>& payload = it->second;
    if (expected_count != static_cast<size_t>(-1) && payload.size() != expected_count) {
        if (err_msg != nullptr) {
            *err_msg = "size mismatch for " + key + ": got=" +
                       std::to_string(payload.size()) + " expected=" +
                       std::to_string(expected_count);
        }
        return false;
    }

    out->assign(payload.size(), 0);
    for (size_t i = 0; i < payload.size(); ++i) {
        if (!parse_i32_token(payload[i], &((*out)[i]))) {
            if (err_msg != nullptr) {
                *err_msg = "invalid int32 payload for " + key + " at index " +
                           std::to_string(i);
            }
            return false;
        }
    }

    return true;
}

inline bool read_i64_array(const RawCaseMap& kv,
                           const std::string& key,
                           size_t expected_count,
                           std::vector<int64_t>* out,
                           std::string* err_msg,
                           bool required = true) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        if (required && err_msg != nullptr) {
            *err_msg = "missing key: " + key;
        }
        return !required;
    }

    const std::vector<std::string>& payload = it->second;
    if (expected_count != static_cast<size_t>(-1) && payload.size() != expected_count) {
        if (err_msg != nullptr) {
            *err_msg = "size mismatch for " + key + ": got=" +
                       std::to_string(payload.size()) + " expected=" +
                       std::to_string(expected_count);
        }
        return false;
    }

    out->assign(payload.size(), 0);
    for (size_t i = 0; i < payload.size(); ++i) {
        if (!parse_i64_token(payload[i], &((*out)[i]))) {
            if (err_msg != nullptr) {
                *err_msg = "invalid int64 payload for " + key + " at index " +
                           std::to_string(i);
            }
            return false;
        }
    }

    return true;
}

inline bool read_float_array(const RawCaseMap& kv,
                             const std::string& key,
                             size_t expected_count,
                             std::vector<float>* out,
                             std::string* err_msg,
                             bool required = true) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        if (required && err_msg != nullptr) {
            *err_msg = "missing key: " + key;
        }
        return !required;
    }

    const std::vector<std::string>& payload = it->second;
    if (expected_count != static_cast<size_t>(-1) && payload.size() != expected_count) {
        if (err_msg != nullptr) {
            *err_msg = "size mismatch for " + key + ": got=" +
                       std::to_string(payload.size()) + " expected=" +
                       std::to_string(expected_count);
        }
        return false;
    }

    out->assign(payload.size(), 0.0f);
    for (size_t i = 0; i < payload.size(); ++i) {
        if (!parse_float_token(payload[i], &((*out)[i]))) {
            if (err_msg != nullptr) {
                *err_msg = "invalid float payload for " + key + " at index " +
                           std::to_string(i);
            }
            return false;
        }
    }

    return true;
}

inline bool read_bool_array(const RawCaseMap& kv,
                            const std::string& key,
                            size_t expected_count,
                            std::vector<bool>* out,
                            std::string* err_msg,
                            bool required = true) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        if (required && err_msg != nullptr) {
            *err_msg = "missing key: " + key;
        }
        return !required;
    }

    const std::vector<std::string>& payload = it->second;
    if (expected_count != static_cast<size_t>(-1) && payload.size() != expected_count) {
        if (err_msg != nullptr) {
            *err_msg = "size mismatch for " + key + ": got=" +
                       std::to_string(payload.size()) + " expected=" +
                       std::to_string(expected_count);
        }
        return false;
    }

    out->assign(payload.size(), false);
    for (size_t i = 0; i < payload.size(); ++i) {
        int v = 0;
        if (!parse_int_token(payload[i], &v)) {
            if (err_msg != nullptr) {
                *err_msg = "invalid bool payload for " + key + " at index " +
                           std::to_string(i);
            }
            return false;
        }
        (*out)[i] = (v != 0);
    }

    return true;
}

inline bool read_scalar_int(const RawCaseMap& kv,
                            const std::string& key,
                            int* out,
                            std::string* err_msg,
                            bool required = true) {
    std::vector<int> tmp;
    if (!read_int_array(kv, key, 1, &tmp, err_msg, required)) {
        return false;
    }
    if (tmp.empty()) {
        return !required;
    }
    *out = tmp[0];
    return true;
}

inline bool read_scalar_i64(const RawCaseMap& kv,
                            const std::string& key,
                            int64_t* out,
                            std::string* err_msg,
                            bool required = true) {
    std::vector<int64_t> tmp;
    if (!read_i64_array(kv, key, 1, &tmp, err_msg, required)) {
        return false;
    }
    if (tmp.empty()) {
        return !required;
    }
    *out = tmp[0];
    return true;
}

} // namespace tb_case_io
} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_TB_CASE_IO_HPP

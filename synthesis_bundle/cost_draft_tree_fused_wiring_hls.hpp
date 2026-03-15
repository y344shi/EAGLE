#ifndef TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
#define TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP

#include <cmath>
#include <cstdint>
#include <limits>

#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_update_hls.hpp"
#include "eagle_tier1_lm_top.hpp"

// Compile-time switch for local RMSNorm gamma usage.
// 0: gamma arrays are provided via function arguments (existing behavior).
// 1: gamma arrays are sourced from eagle4_norm_gamma_2bit.h constants.
#ifndef E4D_USE_LOCAL_NORM_GAMMA
#define E4D_USE_LOCAL_NORM_GAMMA 0
#endif

// Compile-time switch for optional draft prefill stage hardware.
// 0: prefill stage is compiled out to avoid duplicating SLM/LM engines.
// 1: keep prefill stage logic and interfaces active.
#ifndef E4D_ENABLE_PREFILL_STAGE
#define E4D_ENABLE_PREFILL_STAGE 0
#endif

namespace tmac {
namespace hls {

constexpr int kCdtFusedMaxBatch = 128;
constexpr int kCdtFusedMaxNodeTopK = 16;
constexpr int kHlsMaxNodeCount = 4032; // matches Python MAX_Node_Count worst case
constexpr int kHlsHiddenBatch  = 1;    // current iteration uses batch=1 for hidden arrays
constexpr int kEagle4FullVocab = 128256;

// Tripcount policy for HLS synthesis latency estimation.
// Configuration: batch=1, node_top_k=8, depth=16, tree_width=TREE_WIDTH(4).
constexpr int kTcBatch = 1;
constexpr int kTcTopK = 8;
constexpr int kTcDepth = 16;
constexpr int kTcTotalTopK = TREE_WIDTH * kTcTopK;                       // 32
constexpr int kTcMaxNodeCount = 128;
constexpr int kTcMaxVerifyNum = 64;
constexpr int kTcLogitsWidth = 1024;  // typical initial logits candidate count
constexpr int kTcParentAccumSize = kCdtControllerMaxDepth * TREE_WIDTH;  // 256
constexpr int kTcInitScratchSize = kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK; // 2048

// Prefetch / writeback staging loop tripcounts for eagle4_draft_impl.
constexpr int kTcStepFlat    = kTcBatch * TREE_WIDTH;                                           //     4
constexpr int kMaxStepFlat   = kCdtFusedMaxBatch * TREE_WIDTH;                                  //   512
constexpr int kTcTopkFlat    = kTcTotalTopK;                                                    //    32
constexpr int kMaxTopkFlat   = kCdtFusedMaxBatch * TREE_WIDTH * kCdtFusedMaxNodeTopK;           //  8192
constexpr int kTcNodeFlat    = kTcBatch * kTcMaxNodeCount;                                      //   128
constexpr int kMaxNodeFlat   = kCdtFusedMaxBatch * kHlsMaxNodeCount;                            // 516096
constexpr int kTcHidStep     = kTcBatch * TREE_WIDTH * HIDDEN;                                  // 16384
constexpr int kMaxHidStep    = kHlsHiddenBatch * TREE_WIDTH * HIDDEN;                           // 16384
constexpr int kTcHidOut      = kTcBatch * kTcTopK * HIDDEN;                                     // 32768
constexpr int kMaxHidOut     = kHlsHiddenBatch * kCdtFusedMaxNodeTopK * HIDDEN;                 // 65536
constexpr int kTcWorkFlat    = kTcBatch * (kTcMaxVerifyNum + kTcTopK);                          //    72
constexpr int kMaxWorkFlat   = kCdtFusedMaxBatch * (kCdtFusedMaxBatch + kCdtFusedMaxNodeTopK);  // 18432
constexpr int kTcOutputFlat  = kTcBatch * kTcTopK;                                              //     8
constexpr int kMaxOutputFlat = kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK;                        //  2048
constexpr int kTcSortFlat    = kTcBatch * kTcMaxVerifyNum;                                      //    64
constexpr int kMaxSortFlat   = kCdtFusedMaxBatch * kCdtFusedMaxBatch;                           // 16384

// Fused step wiring for one draft-tree layer in HLS:
// 1) score/sort + parent pick + hidden gather,
// 2) cumulative state update (cumu_tokens, prev/next/side_indexs, work/sort_scores).
// KV management uses contiguous HBM ancestor-chain; no tree-mask or controller needed.
void e4d_fused_step(
    // Score inputs
    const float* topk_probas_sampling,      // [batch, tree_width * node_top_k]
    const int64_t* topk_tokens_sampling,    // [batch, tree_width * node_top_k]
    const float* last_layer_scores,         // [batch, tree_width]
    const float* input_hidden_states,       // [batch, tree_width, hidden]
    const int64_t* hot_token_id,            // [vocab]
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Update-state inputs
    const int64_t* topk_indexs_prev,        // [batch, tree_width] global cumu indices

    // Shared dims
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    int verify_num,
    int curr_depth,

    // Capacity dims
    int max_node_count,
    int max_verify_num,

    // Persistent legacy draft state (in/out)
    int64_t* cumu_tokens,                   // [batch, max_node_count]
    float* cumu_scores,                     // [batch, max_node_count]
    int64_t* cumu_deltas,                   // [batch, max_node_count]
    int64_t* prev_indexs,                   // [batch, max_node_count]
    int64_t* next_indexs,                   // [batch, max_node_count]
    int64_t* side_indexs,                   // [batch, max_node_count]
    float* output_scores,                   // [batch, node_top_k]
    int64_t* output_tokens,                 // [batch, node_top_k]
    float* work_scores,                     // [batch, max_verify_num + node_top_k]
    float* sort_scores,                     // [batch, max_verify_num]

    // Fused outputs
    float* output_hidden_states,            // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,            // [batch, node_top_k]

    // Optional debug outputs (can be nullptr)
    float* dbg_curr_layer_scores,           // [batch, tree_width * node_top_k]
    float* dbg_sort_layer_scores,           // [batch, tree_width * node_top_k]
    int64_t* dbg_sort_layer_indices,        // [batch, tree_width * node_top_k]
    int64_t* dbg_parent_indices_in_layer,   // [batch, node_top_k]
    int64_t* dbg_remapped_topk_tokens       // [batch, tree_width * node_top_k]
);

// Backward-compatible wrapper for legacy fused-step TB entrypoint.
inline void cost_draft_tree_fused_step_hls(
    const float* topk_probas_sampling,
    const int64_t* topk_tokens_sampling,
    const float* last_layer_scores,
    const float* input_hidden_states,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    const int64_t* topk_indexs_prev,
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    int verify_num,
    int curr_depth,
    int max_node_count,
    int max_verify_num,
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,
    float* output_hidden_states,
    int64_t* cache_topk_indices,
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens) {
#pragma HLS INLINE
    e4d_fused_step(
        topk_probas_sampling,
        topk_tokens_sampling,
        last_layer_scores,
        input_hidden_states,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        topk_indexs_prev,
        batch_size,
        node_top_k,
        tree_width,
        hidden_size,
        cumu_count,
        verify_num,
        curr_depth,
        max_node_count,
        max_verify_num,
        cumu_tokens,
        cumu_scores,
        cumu_deltas,
        prev_indexs,
        next_indexs,
        side_indexs,
        output_scores,
        output_tokens,
        work_scores,
        sort_scores,
        output_hidden_states,
        cache_topk_indices,
        dbg_curr_layer_scores,
        dbg_sort_layer_scores,
        dbg_sort_layer_indices,
        dbg_parent_indices_in_layer,
        dbg_remapped_topk_tokens);
}

// Fixed tree-width policy helper for orchestrator usage.
// The policy keeps tree_width/verify_num unchanged and never stops early.
struct E4dFixedPolicy {
    void operator()(
        int depth,
        int batch_size,
        int curr_tree_width,
        int node_top_k,
        int max_tree_width,
        int curr_verify_num,
        const float* work_scores,
        int max_verify_num,
        int* next_tree_width,
        int* next_verify_num,
        bool* stop_signal) const {
#pragma HLS INLINE
        (void)depth;
        (void)batch_size;
        (void)node_top_k;
        (void)max_tree_width;
        (void)work_scores;
        (void)max_verify_num;
        if (next_tree_width != nullptr) {
            *next_tree_width = curr_tree_width;
        }
        if (next_verify_num != nullptr) {
            *next_verify_num = curr_verify_num;
        }
        if (stop_signal != nullptr) {
            *stop_signal = false;
        }
    }
};

// Select top-k from logits and output softmax probabilities for those winners.
// If candidate_indices is provided, winner indices are remapped to real vocab token IDs.
void e4d_softmax_topk(
    const float* logits,                 // [batch, logits_width]
    const int64_t* candidate_indices,    // [batch, logits_width] optional
    int batch_size,
    int logits_width,
    int node_top_k,
    float* topk_probas_out,              // [batch, node_top_k]
    int64_t* topk_tokens_out             // [batch, node_top_k]
);

void e4d_copy_frontier(
    const int64_t* frontier_src,  // [batch, max_tree_width]
    int batch_size,
    int max_tree_width,
    int64_t* frontier_dst         // [batch, max_tree_width]
);

// Wire per-layer outputs into the next layer's inputs.
// Selects first next_tree_width entries from node_top_k outputs and
// feeds selected hidden states and global indices back into the next SLM call.
void e4d_prep_next_inputs(
    const float* output_scores,           // [batch, node_top_k]
    const int64_t* output_tokens,         // [batch, node_top_k]
    const float* output_hidden_states,    // [batch, node_top_k, hidden]
    const int64_t* cache_topk_indices,    // [batch, node_top_k]
    int batch_size,
    int node_top_k,
    int hidden_size,
    int next_tree_width,
    int max_tree_width,
    int64_t* next_input_tokens,           // packed [batch, next_tree_width]
    float* next_last_layer_scores,        // packed [batch, next_tree_width]
    float* next_input_hidden_states,      // packed [batch, next_tree_width, hidden]
    int64_t* next_topk_indexs_prev        // packed [batch, next_tree_width]
);

inline void cost_draft_tree_prep_next_inputs_hls(
    const float* output_scores,
    const int64_t* output_tokens,
    const float* output_hidden_states,
    const int64_t* cache_topk_indices,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int next_tree_width,
    int max_tree_width,
    int64_t* next_input_tokens,
    float* next_last_layer_scores,
    float* next_input_hidden_states,
    int64_t* next_topk_indexs_prev) {
#pragma HLS INLINE
    e4d_prep_next_inputs(
        output_scores,
        output_tokens,
        output_hidden_states,
        cache_topk_indices,
        batch_size,
        node_top_k,
        hidden_size,
        next_tree_width,
        max_tree_width,
        next_input_tokens,
        next_last_layer_scores,
        next_input_hidden_states,
        next_topk_indexs_prev);
}

// Run EAGLE4 SLM forward + LM-head top-k for one draft depth.
// The SLM path owns top-k candidate generation; outputs are packed to fused-step layout.
void e4d_slm_topk(
    const float* input_hidden_states,          // packed [batch, tree_width, hidden]
    const float* input_embed_states,           // packed [batch, tree_width, hidden] or nullptr (falls back to hidden)
    int batch_size,
    int tree_width,
    int hidden_size,
    int node_top_k,
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const float rope_cos_vals[HEAD_DIM / 2],
    const float rope_sin_vals[HEAD_DIM / 2],
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    int prefix_len,
    int current_depth,
    const int* parent_indices_per_layer,
    float* reasoning_hidden_states_out,        // packed [batch, tree_width, hidden]
    float* topk_probas_sampling_out,           // packed [batch, tree_width * node_top_k]
    int64_t* topk_tokens_sampling_out          // packed [batch, tree_width * node_top_k]
);

// Multi-layer orchestrator:
//   optional InitialLoop (PyTorch draft_InitialLoop parity):
//     0) from previous-verify logits -> softmax+top-k (or caller-provided initial top-k),
//     1) run one fused tree step with tree_width=1,
//     2) call width policy for depth-0 and seed the first recurrent frontier.
//   recurrent loop per depth:
//     1) run EAGLE4 SLM forward + LM-head top-k (`eagle_tier1_lm_top_eagle4`),
//     2) run one fused tree step (score + update),
//     3) wire fused outputs into next-layer inputs (hidden recurrence + index carry),
//     4) repeat until tree_depth or stop.
//
// KV management uses contiguous HBM ancestor-chain (parent_indices_accum);
// no controller node graph or tree mask is needed.
//
// Policy schedule contract:
//   If use_policy_schedule=true and schedule arrays are non-null, depth-wise policy comes from:
//     policy_next_tree_width[depth], policy_next_verify_num[depth], policy_stop_signal[depth].
//   Otherwise, fixed policy is used: keep curr_tree_width/curr_verify_num and never stop.
void e4d_apply_policy(
    int depth,
    int batch_size,
    int curr_tree_width,
    int node_top_k,
    int max_tree_width,
    int curr_verify_num,
    const float* work_scores,
    int max_verify_num,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int* next_tree_width,
    int* next_verify_num,
    bool* stop_signal);

void e4d_prefill_fc(
    const float* input_hidden_states_3h,   // [batch, 3 * hidden]
    const pack512* fc_weight,              // packed [3*hidden -> hidden]
    const float* fc_scales,                // grouped scales
    int batch_size,
    int hidden_size,
    float* projected_hidden_states);       // [batch, hidden]

void eagle4_draft_impl(
    int tree_depth,
    int curr_depth_start,

    // Optional policy schedule (depth-indexed).
    const int* policy_next_tree_width,   // [policy_depth] optional
    const int* policy_next_verify_num,   // [policy_depth] optional
    const int* policy_stop_signal,       // [policy_depth] optional (0/1)
    int policy_depth,
    bool use_policy_schedule,

    // Step-working buffers (in/out across depths)
    int64_t* step_input_tokens,           // packed [batch, tree_width]
    float* step_input_hidden_states,      // packed [batch, tree_width, hidden]
    float* step_last_layer_scores,        // packed [batch, tree_width]
    int64_t* step_topk_indexs_prev,       // packed [batch, tree_width]
    float* step_topk_probas_sampling,     // packed [batch, tree_width * node_top_k]
    int64_t* step_topk_tokens_sampling,   // packed [batch, tree_width * node_top_k]

    // EAGLE4 SLM + LM-head weights/buffers
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>* rope_cfg_table,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    const uint16_t* draft_embed_tokens_weight, // fp16 [kEagle4FullVocab, hidden]

    // Contiguous KV context
    int prefix_len,
    bool enable_accepted_kv_compact,
    const int64_t* accepted_draft_node_ids, // [accepted_draft_node_count] draft-node IDs from previous verify
    int accepted_draft_node_count,
    int64_t* node_to_hbm_slot,              // [max_node_count], persistent mapping draft_node_id -> hbm_slot_id

    // Hot-token remap config
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Runtime dims (in/out)
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,

    // Capacity dims
    int max_node_count,
    int max_verify_num,
    int max_tree_width,

    // Persistent legacy state
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,

    // Per-step fused outputs / scratch
    float* output_hidden_states,          // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,          // [batch, node_top_k]

    // Optional debug outputs (final executed step contents)
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,

    // Optional loop outputs
    int* executed_depths,
    bool* stopped_early,

    // Optional InitialLoop (disabled by default).
    // If enabled:
    //   - use caller-provided initial_topk_* if present,
    //   - otherwise compute initial_topk_* from initial_logits (+ optional candidate remap).
    // initial_hidden_states must be [batch, hidden] unless prefill-stage is enabled.
    bool enable_initial_loop,// = false,
    const float* initial_logits,// = nullptr,             // [batch, initial_logits_width]
    const int64_t* initial_candidate_indices,// = nullptr,// [batch, initial_logits_width] optional
    int initial_logits_width,// = 0,
    const float* initial_topk_probas,// = nullptr,        // [batch, node_top_k] optional
    const int64_t* initial_topk_tokens,// = nullptr,      // [batch, node_top_k] optional
    const float* initial_hidden_states,// = nullptr       // [batch, hidden]

    // Optional draft-prefill stage (3H -> H projection + one SLM forward).
    // When enabled, initial_topk_* and initial_hidden_states are generated internally
    // from these inputs and override external initial_* tensors.
    bool enable_prefill_stage,// = false
    const float* prefill_input_hidden_states_3h,// = nullptr // [batch, 3 * hidden]
    const float* prefill_input_embed_states,// = nullptr      // [batch, hidden]
    const pack512* prefill_fc_weight,// = nullptr             // packed [3*hidden -> hidden]
    const float* prefill_fc_scales// = nullptr                // grouped scales for prefill_fc_weight
);

// Host-side replay hook for TB parity runs.
// When enabled, eagle4_draft consumes captured recurrent_topk streams
// per depth instead of recomputing SLM top-k in-loop.
void eagle4_draft_set_recurrent_replay(
    const float* recurrent_topk_probas,
    const int64_t* recurrent_topk_tokens,
    int tree_depth,
    int batch_size,
    int max_tree_width,
    int node_top_k);

void eagle4_draft_clear_recurrent_replay();

} // namespace hls
} // namespace tmac

void eagle4_draft(
    int tree_depth,
    int curr_depth_start,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int64_t* step_input_tokens,
    float* step_input_hidden_states,
    float* step_last_layer_scores,
    int64_t* step_topk_indexs_prev,
    float* step_topk_probas_sampling,
    int64_t* step_topk_tokens_sampling,
    const tmac::hls::pack512* w_q,     const float* s_q,
    const tmac::hls::pack512* w_k,     const float* s_k,
    const tmac::hls::pack512* w_v,     const float* s_v,
    const tmac::hls::pack512* w_o,     const float* s_o,
    const tmac::hls::pack512* w_gate,  const float* gate_scales,
    const tmac::hls::pack512* w_up,    const float* up_scales,
    const tmac::hls::pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>* rope_cfg_table,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    const uint16_t* draft_embed_tokens_weight,
    int prefix_len,
    bool enable_accepted_kv_compact,
    const int64_t* accepted_draft_node_ids,
    int accepted_draft_node_count,
    int64_t* node_to_hbm_slot,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,
    float* output_hidden_states,
    int64_t* cache_topk_indices,
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,
    int* executed_depths,
    bool* stopped_early,
    bool enable_initial_loop,
    const float* initial_logits,
    const int64_t* initial_candidate_indices,
    int initial_logits_width,
    const float* initial_topk_probas,
    const int64_t* initial_topk_tokens,
    const float* initial_hidden_states,
    bool enable_prefill_stage,
    const float* prefill_input_hidden_states_3h,
    const float* prefill_input_embed_states,
    const tmac::hls::pack512* prefill_fc_weight,
    const float* prefill_fc_scales);

#endif // TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP

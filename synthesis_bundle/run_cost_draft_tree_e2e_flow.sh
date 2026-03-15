#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="${ROOT_DIR}"
RESULT_ROOT="${ROOT_DIR}/reports/cost_draft_tree_e2e"
CAPTURE_CASE_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL_SRC=""
DUMP_ALL=0
STRICT=0
MULTI_DEPTH_STEPS=3
FEATURE_PROFILES="mid,low,high"
FEATURE_SWEEP=1
REQUIRE_E2E=1
CANDIDATE_SCORE_ATOL="${CANDIDATE_SCORE_ATOL:-1e-4}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Run host-side CostDraftTree E2E validation and capture logs.

Options:
  --case-dir <path>           Case file directory (default: ${CASE_DIR})
  --results-dir <path>        Results root dir (default: ${RESULT_ROOT})
  --capture-case-dir <path>   Optional source dir to copy case files from
  --python <path>             Python executable for dump scripts (default: ${PYTHON_BIN})
  --kernel-src <path>         CUDA kernel source for dump scripts (optional)
  --dump-all                  Attempt full dump_all_cost_draft_tree_cases.sh generation
  --multi-depth-steps <n>     Fused multi-depth synthetic steps (default: ${MULTI_DEPTH_STEPS})
  --feature-profiles <csv>    Profiles to validate (default: ${FEATURE_PROFILES})
  --no-feature-sweep          Validate only mid profile (no low/high)
  --require-e2e               Require E2E case+sidecars for every profile (default)
  --no-require-e2e            Do not require E2E case+sidecars
  --candidate-score-atol <f>  Max score delta across profiles (default: ${CANDIDATE_SCORE_ATOL})
  --strict                    Exit non-zero if any check fails or expected files are missing
  -h, --help                  Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --results-dir) RESULT_ROOT="$2"; shift 2 ;;
    --capture-case-dir) CAPTURE_CASE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --kernel-src) KERNEL_SRC="$2"; shift 2 ;;
    --dump-all) DUMP_ALL=1; shift ;;
    --multi-depth-steps) MULTI_DEPTH_STEPS="$2"; shift 2 ;;
    --feature-profiles) FEATURE_PROFILES="$2"; shift 2 ;;
    --no-feature-sweep) FEATURE_SWEEP=0; shift ;;
    --require-e2e) REQUIRE_E2E=1; shift ;;
    --no-require-e2e) REQUIRE_E2E=0; shift ;;
    --candidate-score-atol) CANDIDATE_SCORE_ATOL="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULT_ROOT}/${timestamp}"
mkdir -p "${RUN_DIR}" "${CASE_DIR}"

SUMMARY_FILE="${RUN_DIR}/summary.txt"
: > "${SUMMARY_FILE}"

log_summary() {
  echo "$*" | tee -a "${SUMMARY_FILE}"
}

run_step() {
  local name="$1"
  shift
  local log_file="${RUN_DIR}/${name}.log"
  echo "[run] ${name}: $*" | tee -a "${SUMMARY_FILE}"
  if "$@" >"${log_file}" 2>&1; then
    log_summary "[pass] ${name}"
    return 0
  else
    local rc=$?
    log_summary "[fail] ${name} (rc=${rc}) log=${log_file}"
    return "${rc}"
  fi
}

EXPECTED_CASE_FILES=(
  "cost_draft_tree_score_case.txt"
  "cost_draft_tree_score_case_hot.txt"
  "cost_draft_tree_update_case.txt"
  "cost_draft_tree_controller_case.txt"
  "cost_draft_tree_fused_wiring_case.txt"
  "cost_draft_tree_multilayer_orchestrator_case.txt"
)
E2E_REQUIRED_FILES=(
  "cost_draft_tree_draft_e2e_case.txt"
  "cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin"
  "cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin"
)

profile_rel_dir() {
  local profile="$1"
  case "${profile}" in
    mid|middle|default|undefined|"")
      echo ""
      ;;
    low|low_intensity|feature_low)
      echo "feature_low"
      ;;
    high|high_intensity|feature_high)
      echo "feature_high"
      ;;
    *)
      echo "feature_${profile}"
      ;;
  esac
}

trim_spaces() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  echo "${s}"
}

build_profiles() {
  if [[ "${FEATURE_SWEEP}" -eq 0 ]]; then
    echo "mid"
    return
  fi
  local out=()
  IFS=',' read -r -a raw <<< "${FEATURE_PROFILES}"
  for item in "${raw[@]}"; do
    item="$(trim_spaces "${item}")"
    [[ -z "${item}" ]] && continue
    out+=("${item}")
  done
  if [[ ${#out[@]} -eq 0 ]]; then
    out=("mid" "low" "high")
  fi
  printf "%s\n" "${out[@]}"
}

sync_profile_cases() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  for f in "${EXPECTED_CASE_FILES[@]}" "${E2E_REQUIRED_FILES[@]}" "cost_draft_tree_case_manifest.txt"; do
    if [[ -f "${src_dir}/${f}" ]]; then
      cp -f "${src_dir}/${f}" "${dst_dir}/${f}"
      log_summary "[copied] ${dst_dir}/${f}"
    fi
  done
  for d in "cpmcu_tensors" "packed_all"; do
    if [[ -d "${src_dir}/${d}" ]]; then
      mkdir -p "${dst_dir}/${d}"
      cp -af "${src_dir}/${d}/." "${dst_dir}/${d}/"
      log_summary "[copied] ${dst_dir}/${d}/"
    fi
  done
}

compare_e2e_profiles() {
  local score_atol="$1"
  shift
  local -a case_refs=("$@")
  if [[ ${#case_refs[@]} -lt 2 ]]; then
    return 0
  fi
  run_step "profile_candidate_consistency" "${PYTHON_BIN}" - "${score_atol}" "${case_refs[@]}" <<'PY'
import sys
from pathlib import Path

def parse_case(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid line: {line}")
            key = parts[0]
            count = int(parts[1])
            payload = parts[2:]
            if len(payload) != count:
                raise ValueError(f"{path}: payload mismatch for {key}")
            out[key] = payload
    return out

def as_ints(v):
    return [int(x) for x in v]

def as_floats(v):
    return [float(x) for x in v]

def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0

atol = float(sys.argv[1])
pairs = [arg.split("=", 1) for arg in sys.argv[2:]]
if len(pairs) < 2:
    raise SystemExit(0)

parsed = {}
for profile, p in pairs:
    path = Path(p)
    kv = parse_case(path)
    required = ["expected_output_tokens", "expected_output_scores", "recurrent_topk_tokens"]
    for key in required:
        if key not in kv:
            raise SystemExit(f"[error] {profile}: missing key {key} in {path}")
    parsed[profile] = {
        "path": path,
        "expected_output_tokens": as_ints(kv["expected_output_tokens"]),
        "expected_output_scores": as_floats(kv["expected_output_scores"]),
        "recurrent_topk_tokens": as_ints(kv["recurrent_topk_tokens"]),
    }

base_profile = pairs[0][0]
base = parsed[base_profile]
for profile, _ in pairs[1:]:
    curr = parsed[profile]
    if curr["expected_output_tokens"] != base["expected_output_tokens"]:
        raise SystemExit(
            f"[error] expected_output_tokens mismatch: {base_profile} vs {profile}"
        )
    if curr["recurrent_topk_tokens"] != base["recurrent_topk_tokens"]:
        raise SystemExit(
            f"[error] recurrent_topk_tokens mismatch: {base_profile} vs {profile}"
        )
    if len(curr["expected_output_scores"]) != len(base["expected_output_scores"]):
        raise SystemExit(
            f"[error] expected_output_scores length mismatch: {base_profile} vs {profile}"
        )
    delta = max_abs_diff(base["expected_output_scores"], curr["expected_output_scores"])
    if delta > atol:
        raise SystemExit(
            f"[error] expected_output_scores drift {delta:.6g} exceeds atol={atol} for {base_profile} vs {profile}"
        )
    print(f"[ok] {base_profile} vs {profile}: tokens+recurrent match, max_score_delta={delta:.6g}")

print("[PASS] cross-profile candidate consistency check passed")
PY
}

OVERALL_FAIL=0
MISSING_COUNT=0
mapfile -t PROFILES < <(build_profiles)
if [[ ${#PROFILES[@]} -eq 0 ]]; then
  PROFILES=("mid")
fi

{
  echo "timestamp=${timestamp}"
  echo "root_dir=${ROOT_DIR}"
  echo "case_dir=${CASE_DIR}"
  echo "results_dir=${RUN_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "kernel_src=${KERNEL_SRC:-<none>}"
  echo "capture_case_dir=${CAPTURE_CASE_DIR:-<none>}"
  echo "feature_profiles=${PROFILES[*]}"
  echo "feature_sweep=${FEATURE_SWEEP}"
  echo "require_e2e=${REQUIRE_E2E}"
  echo "candidate_score_atol=${CANDIDATE_SCORE_ATOL}"
  echo "strict=${STRICT}"
  echo "multi_depth_steps=${MULTI_DEPTH_STEPS}"
  git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/'
  git -C "${ROOT_DIR}" status --short 2>/dev/null | sed 's/^/git_status=/'
} > "${RUN_DIR}/env.txt"

if [[ -n "${CAPTURE_CASE_DIR}" ]]; then
  log_summary "[info] syncing cases from ${CAPTURE_CASE_DIR}"
  for profile in "${PROFILES[@]}"; do
    rel="$(profile_rel_dir "${profile}")"
    src_dir="${CAPTURE_CASE_DIR}"
    dst_dir="${CASE_DIR}"
    if [[ -n "${rel}" ]]; then
      src_dir="${CAPTURE_CASE_DIR}/${rel}"
      dst_dir="${CASE_DIR}/${rel}"
    fi
    if [[ -d "${src_dir}" ]]; then
      sync_profile_cases "${src_dir}" "${dst_dir}"
    else
      log_summary "[warn] capture profile dir missing: ${src_dir}"
    fi
  done
fi

if [[ ! -s "${CASE_DIR}/cost_draft_tree_controller_case.txt" ]]; then
  if run_step "dump_controller_case" "${PYTHON_BIN}" "${ROOT_DIR}/dump_cost_draft_tree_controller_case.py" --output "${CASE_DIR}/cost_draft_tree_controller_case.txt"; then
    :
  else
    OVERALL_FAIL=1
  fi
fi

if [[ "${DUMP_ALL}" -eq 1 ]]; then
  for profile in "${PROFILES[@]}"; do
    rel="$(profile_rel_dir "${profile}")"
    profile_case_dir="${CASE_DIR}"
    if [[ -n "${rel}" ]]; then
      profile_case_dir="${CASE_DIR}/${rel}"
    fi
    mkdir -p "${profile_case_dir}"

    dump_cmd=(bash "${ROOT_DIR}/dump_all_cost_draft_tree_cases.sh" --python "${PYTHON_BIN}" --output-dir "${profile_case_dir}")
    if [[ -n "${KERNEL_SRC}" ]]; then
      dump_cmd+=(--kernel-src "${KERNEL_SRC}")
    fi
    if [[ -s "${profile_case_dir}/cost_draft_tree_draft_e2e_case.txt" ]]; then
      dump_cmd+=(--e2e-case "${profile_case_dir}/cost_draft_tree_draft_e2e_case.txt")
    fi
    if run_step "dump_all_cases_${profile}" "${dump_cmd[@]}"; then
      :
    elif [[ "${STRICT}" -eq 1 ]]; then
      OVERALL_FAIL=1
    fi
  done
fi

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -I.)
HOST_LINK_INCLUDES=(
  -include eagle4_lm_head_hls.cpp
  -include eagle_tier1_lm_top.cpp
  -include eagle_tier1_top.cpp
  -include cost_draft_tree_fused_wiring_hls.cpp
)

if run_step "compile_fused_tb_host" "${CXX}" "${CXXFLAGS[@]}" "${HOST_LINK_INCLUDES[@]}" "${ROOT_DIR}/cost_draft_tree_fused_wiring_tb.cpp" -O2 -o /tmp/cdt_fused_wiring_tb_host; then
  run_step "fused_synthetic" /tmp/cdt_fused_wiring_tb_host || OVERALL_FAIL=1
  run_step "fused_synthetic_multidepth" /tmp/cdt_fused_wiring_tb_host --multi-depth-steps "${MULTI_DEPTH_STEPS}" || OVERALL_FAIL=1
else
  OVERALL_FAIL=1
fi

E2E_CASE_REFS=()
for profile in "${PROFILES[@]}"; do
  rel="$(profile_rel_dir "${profile}")"
  profile_case_dir="${CASE_DIR}"
  if [[ -n "${rel}" ]]; then
    profile_case_dir="${CASE_DIR}/${rel}"
  fi
  mkdir -p "${profile_case_dir}"
  log_summary "[profile] ${profile} -> ${profile_case_dir}"

  for f in "${EXPECTED_CASE_FILES[@]}"; do
    if [[ -s "${profile_case_dir}/${f}" ]]; then
      log_summary "[present:${profile}] ${f}"
    else
      log_summary "[missing:${profile}] ${f}"
      MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
  done

  if [[ "${REQUIRE_E2E}" -eq 1 ]]; then
    for f in "${E2E_REQUIRED_FILES[@]}"; do
      if [[ -s "${profile_case_dir}/${f}" ]]; then
        log_summary "[present:${profile}] ${f}"
      else
        log_summary "[missing:${profile}] ${f}"
        MISSING_COUNT=$((MISSING_COUNT + 1))
      fi
    done
  fi

  e2e_case="${profile_case_dir}/cost_draft_tree_draft_e2e_case.txt"
  if [[ -s "${e2e_case}" ]]; then
    E2E_CASE_REFS+=("${profile}=${e2e_case}")
    run_step "regen_orchestrator_from_e2e_${profile}" \
      "${PYTHON_BIN}" "${ROOT_DIR}/dump_cost_draft_tree_multilayer_orchestrator_case.py" \
      --e2e-case "${e2e_case}" \
      --output "${profile_case_dir}/cost_draft_tree_multilayer_orchestrator_case.txt" || OVERALL_FAIL=1
  elif [[ "${REQUIRE_E2E}" -eq 1 ]]; then
    OVERALL_FAIL=1
  fi

  if [[ -s "${profile_case_dir}/cost_draft_tree_fused_wiring_case.txt" ]]; then
    run_step "fused_casefile_${profile}" /tmp/cdt_fused_wiring_tb_host --case-file "${profile_case_dir}/cost_draft_tree_fused_wiring_case.txt" || OVERALL_FAIL=1
  fi

  check_args=()
  if [[ "${REQUIRE_E2E}" -eq 1 ]]; then
    check_args+=(--require-e2e)
  fi
  if run_step "golden_check_full_${profile}" bash "${ROOT_DIR}/check_cost_draft_tree_goldens.sh" "${check_args[@]}" "${profile_case_dir}"; then
    :
  elif [[ "${STRICT}" -eq 1 ]]; then
    OVERALL_FAIL=1
  fi

  if run_step "golden_check_dryrun_${profile}" bash "${ROOT_DIR}/check_cost_draft_tree_goldens.sh" --dry-run "${check_args[@]}" "${profile_case_dir}"; then
    :
  elif [[ "${STRICT}" -eq 1 ]]; then
    OVERALL_FAIL=1
  fi
done

if ! compare_e2e_profiles "${CANDIDATE_SCORE_ATOL}" "${E2E_CASE_REFS[@]}"; then
  OVERALL_FAIL=1
fi

if [[ "${STRICT}" -eq 1 && "${MISSING_COUNT}" -gt 0 ]]; then
  OVERALL_FAIL=1
fi

if [[ "${OVERALL_FAIL}" -ne 0 ]]; then
  log_summary "[result] FAIL run_dir=${RUN_DIR}"
  exit 1
fi

log_summary "[result] PASS run_dir=${RUN_DIR}"

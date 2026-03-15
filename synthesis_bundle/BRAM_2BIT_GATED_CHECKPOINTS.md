# BRAM + 2-bit GPTQ Migration Checkpoints

Branch: `codex/2bit-bram-migration-eagle4`
Top: `eagle4_draft_packed_32pc`

## Phase 1: 2-bit authority + packing/dequant migration
Status: PASS

Changes:
- Made `TMAC_WEIGHT_BITS` authoritative in `tmac_utils.hpp` (default `2`, validated `2|4`).
- Replaced pack-count `/128` assumptions with `kPack512WeightElems` in packed layout and key testbenches.
- Migrated LM-head candidate dequant path from hardcoded int4 assumptions to bitwidth-driven unpack/dequant (`kLmTcQpackFactor`, `kLmDefaultZeroPoint`, dynamic qzero unpack).
- Switched prefill FC projection to INT2 kernel when `TMAC_WEIGHT_BITS==2`.

Checkpoint commands:
- `deep_pipeline_lutmac_tb --smoke` -> PASS
- `test_eagle4_lm_head --smoke` -> PASS
- `test_eagle_top_eagle4 --smoke` -> PASS

## Phase 2: BRAM-first projection tile staging
Status: PASS

Changes:
- Added `TMAC_BIND_PROJECTION_TILE_TO_BRAM` (default `1`) in `deep_pipeline_lutmac.hpp`.
- Bound activation/output/tile weight/tile scale scratch arrays to BRAM in:
  - `dense_projection_production_scaled`
  - `dense_projection_production_scaled_batched`
  - `dense_projection_production_scaled_w2`
  - `dense_projection_production_scaled_batched_w2`
- Set HLS config cflags:
  - `-DTMAC_WEIGHT_BITS=2`
  - `-DTMAC_BIND_PROJECTION_TILE_TO_BRAM=1`

Checkpoint commands:
- `deep_pipeline_lutmac_tb --smoke` -> PASS
- `test_eagle_top_eagle4 --smoke` -> PASS

## Phase 3: Full smoke gate
Status: PASS

Checkpoint commands:
- `test_eagle4_lm_head --smoke` -> PASS
- `test_eagle_top_eagle4 --smoke` -> PASS
- `test_eagle_tier1_lm_top_eagle4 --smoke` -> PASS (compile/path smoke)

## Phase 4: HLS synthesis
Status: INCOMPLETE (manual stop)

Run command:
- `v++ -c --mode hls --config /home/y344shi/workspace/amdv80/hls_eagle4_synthesis/vitis_hls_workspace/eagle4_end_to_end/hls_config.cfg --work_dir /home/y344shi/workspace/amdv80/hls_eagle4_synthesis/vitis_hls_workspace/eagle4_end_to_end`

Observed behavior:
- Reached deep `csynth` scheduling/binding stages for multiple `dense_projection_production_scaled_batched_w2` clones.
- No final `csynth.rpt`/`csynth.xml` generated yet in `hls/syn/report`.

Resume guidance:
- Re-run the same `v++` command and wait for completion, then parse:
  - `.../hls/syn/report/csynth.rpt`
  - `.../hls/syn/report/csynth.xml`

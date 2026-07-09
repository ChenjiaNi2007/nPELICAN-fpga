# Per-stage function split (resource attribution build)

`firmware/nPELICAN_split.cpp` is a restructured copy of `firmware/nPELICAN.cpp`
whose only purpose is **resource/latency attribution**: each datapath stage is a
separate function with `#pragma HLS INLINE off`, so the csynth report breaks
LUT/FF/DSP/latency down per stage instead of one flat top-level number. Until
now we could only guess (from operator names like `mul_18s_18s`) which part of
the design was consuming what.

The monolith stays the default and the reference. **Exactly one** of
{`nPELICAN.cpp`, `nPELICAN_split.cpp`} goes into any build — they define the
same top function `nPELICAN` (and the same csim hooks), so they can never be
linked together. If the split build turns out more expensive, ship the
monolith; the split build still tells you *where* the resources go.

## Stage map

Boundaries sit exactly on the quantization points, so every array crossing a
function boundary already has its intended narrow per-stage type and is
completely partitioned on both sides (wires, never BRAM).

| function | contents | output (type) | expected dominant cost |
|---|---|---|---|
| `np_dots` | p1/beam prep + symmetric dot4 | `dots[484]` (`dot_t`) | DSP (≈253 upper-triangle dots) |
| `np_bn1` | BN1 affine, mean-folded, masked | `batch1[484]` (`bn1out_t`) | DSP/LUT (253 wide mults) |
| `np_agg2to2` | raw Σ + normalize-late (×invnave/invnave2) | `jmass`, `jdotp[22]` (`t2_t`) | LUT adder trees + 23 norm mults |
| `np_eq2to2` | basis `T` (internal, pure wiring) + 2→2 MAC + ReLU quant | `Tp_q[22][22][2]` (`relu_t`) | MAC DSP/LUT |
| `np_agg2to0` | 2→0 sum/trace + collapsed BN2 affine + normalize | `R[2][2]` (`t0_t`) | adder trees, 8 wide mults |
| `np_out2to0` | 2→0 dense + bias | `Rp[1]` (`mac0_t`) | tiny |

Kept in the **top**: the nobj remap + `nobjmask` build (comparators/wiring,
shared by four stages), the csim-only dots-override/stage-dump hooks, and the
final `(result_t)` output cast. `T` and `Tr` are reconstructed inside the
top's csim-only dump block (identical expressions) since they are no longer
materialized in top scope.

## How to build / read the report

Remote Vitis box:

```bash
vitis_hls -f build_prj.tcl split=1 reset=1        # csim + csynth
```

This uses its own project dir `nPELICAN_split_prj` (the monolith's
`nPELICAN_prj` reports are never touched). `split=1` supports `csim`/`synth`
only; `cosim`/`validation`/`export`/`vsynth` are forced off (their tcl plumbing
assumes the monolith project dir — extend if ever needed).

The resource-lever build flags apply to BOTH builds (`RESOURCE_REDUCTION_LEVERS.md`
Levers 5/6): `const_beams=1` (constant beam spurions, `-DNPELICAN_CONST_BEAMS`)
and `mac_dsp=1` (2→2 MAC mults bound to DSP48, `-DNPELICAN_MAC_DSP`). Use
`reset=1` when toggling flags so stale cflags can't leak from an existing
project dir; run lever experiments on the split build first so the delta is
attributed to the right stage.

Per-stage numbers: `nPELICAN_split_prj/solution/syn/report/csynth.rpt` —
the "Utilization Estimates > Detail > Instance" table lists one
`grp_np_<stage>_fu_*` row per stage with its LUT/FF/DSP, and the Latency
section gives per-instance latency. Compare against the monolith totals from
`nPELICAN_prj` built from the same `weights.h`.

Local bit-exactness gate (no Vitis needed):

```bash
./build_local.sh       -DRUN_GOLDEN_GATE   # → tb_local        (monolith)
./build_local.sh split -DRUN_GOLDEN_GATE   # → tb_local_split  (split)
```

Run both and diff `tb_data/golden_fw_results.log`, `tb_data/fw_stage_dump.txt`,
and the printed gate summaries between the two: they must be **byte-identical**
(the split is pure code motion in exact fixed-point arithmetic — any diff at
all means the split drifted from the monolith and the monolith wins).
Verified identical on 2026-07-03 (200 golden events + 10k legacy flow).

## Stage isolation — one-boundary-at-a-time marginal costs

The full split measures every stage in a degraded context: all six boundaries
exist at once, so per-stage numbers include lost cross-boundary sharing (e.g.
np_eq2to2 reported 529 DSP in the full split vs ~335 total non-dot DSP implied
by the monolith), and the aggregate overhead (+232 DSP / +45k FF / +8 cyc on
2026-07-07) cannot be attributed to a specific boundary.

`split_only=<stage>` fixes both: ONLY the named stage stays a function; the
other five are force-inlined back into the top (preprocessor-selected
`INLINE off` vs `INLINE` — the code, and therefore csim output, is identical
in every configuration; verified byte-identical for all six on 2026-07-09).
Each run then yields two clean numbers against `reports/csynth_monolith.rpt`:

- the stage's `grp_np_<stage>` row = its **true marginal cost** with the rest
  of the design still globally optimized (the number to quote per component);
- run total − monolith baseline = that **one boundary's overhead**
  (boundary registers + lost sharing + added latency cycles).

```bash
# per stage (own project dir nPELICAN_split_<stage>_prj — reports accumulate):
vitis_hls -f build_prj.tcl reset=1 csim=0 cosim=0 validation=0 export=0 vsynth=0 split_only=dots
# stages: dots  bn1  agg2to2  eq2to2  agg2to0  out2to0
```

Run these WITHOUT `const_beams`/`mac_dsp` so they stay comparable to the
monolith baseline. Priority if synth time is scarce — the 3-run reduced set
{dots, eq2to2, bn1} covers ~90% of LUT and ~97% of DSP; the aggregation and
output stages are refinements. Record results in the isolation table in
`resource_log.md`. (This is the cumulative-splitting idea from review,
strengthened: isolating one stage per run makes each measurement independent
of the order in which boundaries are introduced.)

## Resource-overhead watchlist (why splitting can cost resources)

The known mechanisms by which "same design, more functions" inflates HLS
resources, and how this split guards against each — check these in the report
before trusting a regression:

1. **Boundary arrays falling out of complete partitioning** → BRAM/muxes.
   Guard: every array argument carries `ARRAY_PARTITION complete dim=0` inside
   the callee, matching the caller-side pragma.
2. **Wide pure-wiring tensors becoming function ports** → port/FF bloat.
   Guard: `T` (2904 × t2_t) is internal to `np_eq2to2`; boundaries carry only
   quantization-point data.
3. **Per-instance control logic / start-propagation FF** under the top
   `PIPELINE II=1`. Unavoidable but small (order tens of FF per instance);
   each stage carries its own `PIPELINE II=1` so it schedules inside the top
   pipeline. A few cycles of extra pipeline latency are possible — check II=1
   still holds and note the latency delta in `resource_log.md`.
4. **Lost cross-stage optimization** (CSE/bitwidth propagation across the
   boundary). The boundaries are quantization points, where values are rounded
   to their learned grid anyway, so there is nothing to share across them by
   construction.

If a stage unexpectedly shows as *inlined anyway* (no `grp_np_*` instance in
the report), the tool version force-inlined under the pipeline — the
`INLINE off` pragmas are the override; check the csynth log for inline
messages before believing a flat report.

## Keeping the two files in sync

Any datapath change made to `nPELICAN.cpp` must be mirrored into the matching
stage of `nPELICAN_split.cpp` (or the split file regenerated from it), then the
local byte-identical gate re-run. The split file is a tool, not a fork: the
monolith is always the reference implementation.

# n-hidden capacity sweep

Quantifies how nPELICAN **model size** trades off against **accuracy** and **FPGA
resources**. The single knob is `--n-hidden N` (channels between Eq2to2 and Eq2to0);
the firmware mirrors it via `#define NHIDDEN` in `firmware/nPELICAN.h`.

## Layout assumption

This dir lives in `nPELICAN-fpga/`; it expects `PELICAN-nano/` as a **sibling**
(`../PELICAN-nano`). Override with `PN=/path/to/PELICAN-nano`. Use the QAT venv's
python via `PY=` so torch/Brevitas import.

## Run

```bash
cd nPELICAN-fpga/sweep
PY=../../PELICAN-nano/.venv/bin/python NS="1 2 3 4 6" EPOCHS=8 ./sweep_nhidden.sh
```

Per N: train QAT → count params + read AUC/accuracy → set `NHIDDEN` → export
`weights.h`/`types_generated.h` → regenerate golden vectors → local csim gate →
Vitis csynth (if `vitis_hls` on PATH) → append a row → render figures.

Env knobs: `NS`, `EPOCHS` (cos LR needs ≥8; `DECAY=flat` allows ≥5), `GOLDEN_N`,
`DATADIR` (h5 dir globbed for train/valid/test, **rel. to PELICAN-nano**; default
`data/sample_data`), `DO_SYNTH` (`auto`/`yes`/`no`), `PN`, `PY`, plus the pinned
quant/recipe knobs below.

### Pinned quant + training config (only `--n-hidden` varies)

The bit widths and regularization are **fixed** to the production recipe
(`PELICAN-nano/scripts/sweep_pmu_width.sh`) so every sweep point differs only in
model size. Defaults: `WBITS=6 ABITS=6 IBITS=6 PMU=12`, `SEED=42`,
`DROP=0.05 DROP_OUT=0.05 WD=0.005 NOBJ=20 NOBJ_AVG=49 BATCH=256`,
`DEVICE="--no-reproducible"`.

> **Do not** drop the bit-width flags: argparse defaults are `8/8/8` with
> `pmu=None`, and `pmu=None` leaves the momenta **un-quantized (float)** — which
> does not match the firmware `input_t`. Override e.g. `PMU=9` to move the momentum
> grid, but keep every knob identical across the whole sweep.

> The trainer globs `--datadir/*.h5` and picks files by `train`/`valid`/`test` in
> the name — it does **not** recurse. Point `DATADIR` at the dir that directly holds
> the `.h5` files (the repo default `data/` is one level too high).

## Outputs (written here)

- `sweep_results.csv` — one row per N: `n_hidden,params,accuracy,AUC,BgRej03,csim,LUT,FF,DSP,BRAM,lat_cycles,lat_ns,II`
- `sweep_accuracy.png` — params → AUC & accuracy
- `sweep_resources.png` — params → LUT/FF/DSP (only rows with csynth data)
- `sweep_pareto.png` — AUC vs DSP, annotated with `h=N`
- per-N csynth reports copied to `../reports/csynth_nhid<N>.rpt`

## Resources: parallel csynth over the trained checkpoints

`sweep_nhidden.sh` completes the accuracy half anywhere; run resources separately on
the Vitis box with **`synth_sweep.sh`**. It stages an isolated per-N tree under
`synth_builds/nhid<N>/` (each with the correct `firmware/weights/` and `NHIDDEN`),
launches `vitis_hls` concurrently, then backfills `sweep_results.csv` and re-plots.

Concurrency safety: each N has its own tree, so per-tree `nPELICAN_prj/` and
`vivado_synth.rpt` never collide. Each job copies its own reports into unique
`reports/{csynth,vsynth}_nhid<N>.rpt` the moment it finishes (stale copies cleared at
stage time), and the CSV backfill runs serially after `wait` — one writer, no races.

```bash
# on the Vitis box, after sweep_nhidden.sh has trained the checkpoints:
export XILINX_HLS=/tools/Xilinx/Vitis_HLS/2023.2
export XILINX_VIVADO=/tools/Xilinx/Vivado/2023.2
export PATH=$XILINX_HLS/bin:$XILINX_VIVADO/bin:$PATH

PY=../../PELICAN-nano/.venv/bin/python NS="1 2 3 4 6" ./synth_sweep.sh
# knobs: VSYNTH=1 (default; adds Vivado post-synth cols), JOBS=3 (throttle concurrency)
```

Each run uses `reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=$VSYNTH`.
`VSYNTH=1` (default) additionally runs `vivado` post-synthesis and adds real
**`vLUT,vFF,vDSP,vBRAM`** columns — these are the numbers to report (the HLS
`LUT/FF/...` columns are pre-implementation *estimates*, typically ~3× high on LUTs).
The plots automatically prefer the `v*` columns when present. Needs `vivado` on PATH;
set `VSYNTH=0` for HLS estimates only (faster). Does **not** retrain — reuses the
`model/nhid<N>_best.pt` checkpoints.

Manual single-N backfill is also available:

```bash
python sweep_tools.py backfill --csv sweep_results.csv --n 4 \
    --rpt synth_builds/nhid4/nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt
python sweep_tools.py backfill-vsynth --csv sweep_results.csv --n 4 \
    --rpt synth_builds/nhid4/vivado_synth.rpt
python sweep_plot.py --csv sweep_results.csv
```

### The firmware/weights gotcha (why csim can read `fail`)

The firmware includes `"weights/weights.h"` **relative to `firmware/`**, i.e.
`firmware/weights/weights.h` — NOT the repo-root `weights/`. `model_loader.py`'s
default `--out weights/weights.h` writes to the wrong place, so the firmware keeps
compiling stale weights while `NHIDDEN` changes → out-of-bounds indexing → `csim=fail`
and invalid synthesis. Both `sweep_nhidden.sh` and `synth_sweep.sh` pass
`--out .../firmware/weights/weights.h` to fix this. If you export by hand, do the same.

## Notes

- Clean single-variable sweep: only `--n-hidden` varies; quant bit-widths, seed,
  epochs, and data are held fixed so deltas are attributable to capacity alone.
- `csim=pass` means firmware logits matched the PyTorch quant golden (`GOLDEN GATE:
  PASS`). A row with `fail`/`ran` still has valid *accuracy*; treat its resource
  numbers as not-yet-bit-exact-validated.
- `AUC` is the headline accuracy metric; `BgRej03` is background rejection at 0.3
  signal efficiency.

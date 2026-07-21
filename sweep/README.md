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
`DO_SYNTH` (`auto`/`yes`/`no`), `PN`, `PY`.

## Outputs (written here)

- `sweep_results.csv` — one row per N: `n_hidden,params,accuracy,AUC,BgRej03,csim,LUT,FF,DSP,BRAM,lat_cycles,lat_ns,II`
- `sweep_accuracy.png` — params → AUC & accuracy
- `sweep_resources.png` — params → LUT/FF/DSP (only rows with csynth data)
- `sweep_pareto.png` — AUC vs DSP, annotated with `h=N`
- per-N csynth reports copied to `../reports/csynth_nhid<N>.rpt`

## Split accuracy (local) from resources (remote Vitis)

If the box has no Vitis, `DO_SYNTH=auto` completes the accuracy half and stages the
firmware per N. Later, on the Vitis box, run `vitis_hls -f build_prj.tcl` for each
staged N and backfill:

```bash
python sweep_tools.py parse-csynth --rpt <path>/nPELICAN_csynth.rpt
python sweep_plot.py --csv sweep_results.csv   # re-render with resources
```

## Notes

- Clean single-variable sweep: only `--n-hidden` varies; quant bit-widths, seed,
  epochs, and data are held fixed so deltas are attributable to capacity alone.
- `csim=pass` means firmware logits matched the PyTorch quant golden (`GOLDEN GATE:
  PASS`). A row with `fail`/`ran` still has valid *accuracy*; treat its resource
  numbers as not-yet-bit-exact-validated.
- `AUC` is the headline accuracy metric; `BgRej03` is background rejection at 0.3
  signal efficiency.

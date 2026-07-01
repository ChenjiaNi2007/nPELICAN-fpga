# HANDOFF — DeepSet vs nanoPELICAN comparison (for a fresh Claude session)

**Main goal right now:** retrain the DeepSet on `data/toptag`, then produce the equivariance
overlay plots (and the C6 ROC overlay) with the *newly trained* DeepSet against nanoPELICAN —
**both models on the same dataset**. Everything below is the state needed to finish that.

---

## 0. The one big thing to understand first (why we're redoing the DeepSet)

The comparison was accidentally **apples-to-oranges on the dataset axis**, and this was the
root cause of a long-running confusion about nanoPELICAN's AUC (0.9529 vs ~0.70):

- **nanoPELICAN** was trained on `PELICAN-nano/data/toptag` (~1.21M jets, **200 constituents**,
  ~20 real particles/jet). Its headline **AUC 0.9529** is on that set's test split
  (`nPELICAN-fpga/tb_data/full_pmu_test.dat`, 50k jets).
- **DeepSet** was (originally) trained + tested on `PELICAN-nano/data/sample_data`
  (39,936 test jets, **20 constituents**, ~12 real particles/jet). The equivariance study
  also uses `sample_data/test.h5`.
- On `sample_data`, nanoPELICAN scores only **~0.70** — it's **out-of-distribution** there.
  Proven empirically on the dev Mac: the *identical* 6:6:6 firmware (dot_t=6-bit, input_t=18)
  gives **0.9529 on `full_pmu_test.dat`** and **0.7028 on the sample_data canonical set** —
  same weights, same types, only the data differs.

**`--max-input-bits` / fixed-point precision was a RED HERRING.** It's a Lever-2 *cap* on
`input_t` (shaves 24→18 bits); measured effect on AUC ≈ nil (0.7028 → 0.7004). The AUC gap is
100% the dataset.

**Decision (made with the user): put BOTH models on `data/toptag`** — the standard benchmark,
nanoPELICAN's home, and where its whole equivariance/resource-reduction body of work already
lives. Moving the (cheap, new) DeepSet is far less costly than re-doing all of nanoPELICAN.

---

## 1. Repos / branches / layout

Sibling repos. **Pod** (training/synthesis): `~/ClaudeInProgress/{l1-jet-id, nPELICAN-fpga,
PELICAN-nano}`. **Dev Mac** (has a working torch/brevitas venv + g++, used for empirical
firmware checks): `~/Desktop/Rankin Research/{...}`.

- `l1-jet-id` branch **`toptag-comparison`** (fork ChenjiaNi2007/l1-jet-id) — DeepSet code.
- `nPELICAN-fpga` branch **`deepset-equivariance`** — equivariance overlay + this `comparison/`.
- Both are **feature branches, NOT merged to main.** Check them out explicitly.

Pod dataset: `PELICAN-nano/data/toptag/{train,test}.h5` — verified present, keys
`Pmu (njets,200,4)` in (E,px,py,pz) + `is_signal` (frac 0.5). `data/toptag` is NOT on the dev
Mac (only `sample_data`).

---

## 2. What's DONE (committed + pushed)

**l1-jet-id `toptag-comparison`:**
- Phase C DeepSet adapter `fast_jetclass/data/toptag_data.py` — reads PELICAN-nano h5, converts
  4-mom → jet-relative (pT,eta_rel,phi_rel); **now selects leading-`nconst` by pT** from the
  200-wide toptag Pmu (`_select_leading_pt`, matches nanoPELICAN's `--nobj` cap) [a7dc14b].
- `util.import_data` dispatch (`dataset_type: "toptag"`); `deepsets_test --h5_path` override;
  `scripts/roc_overlay.py` (C6); plot-label fix; softmax strip in `synthesize.py` [0c7bbaa].
- Config trio `scripts/configs/deepsets_8bit_20const_toptag/` (+ 8/16/32) **now point at
  `data/toptag/train.h5`**, `root: ../data/jetid_toptagset` (fresh norm cache) [e702410].
- `environment.yml`: numpy pinned 1.26 (TF 2.14 compat) [a7fc443]. Note: a server-side commit
  had switched TF to `tensorflow-cpu`; that's being replaced by GPU TF (see §4).

**nPELICAN-fpga `deepset-equivariance`:**
- `equivariance/run_sweep_deepset.py` — DeepSet equivariance evaluator (loads canonical boosted
  4-mom, reuses train norm pkl, runs the DeepSet, emits one logit/manifest-row). Loads the
  QKeras model via `qkeras.utils.load_qmodel`.
- `equivariance/overlay_deepset.py` — overlays the DeepSet curve on the **committed**
  nanoPELICAN `aggregates_{mode}.csv` (no firmware rerun needed); also writes
  `results/aggregates_deepset.csv` [09b1ea5]. Outputs `results/plots_deepset_overlay/*.png`.
- `equivariance/run_sweep.py` — skips missing checkpoints instead of aborting [b43eafb].
- `equivariance/config.yaml` — added `deepset` model entry; checkpoint paths now point at
  vendored `equivariance/checkpoints/*.pt` (7 QAT checkpoints, ~60KB each) [b2a6ef8].
- `project.tcl` — FPGA part set to `xcvu13p-flga2577-2-e` (D2, matches DeepSet synth) [7777c74].
- `comparison/` — `REPORT.md`, `deepset_resources.csv`, `npelican_resources.csv`,
  `roc_summary.csv` (all from the OLD sample_data run — see §5, need refresh on toptag).

**Resource numbers already collected (csynth, xcvu13p-2, 5ns), STILL VALID (arch, not data):**
- DeepSet 8-bit N=20: DSP 124, LUT 665,981, FF 343,142, latency 144cyc/0.72µs, II=40.
- nanoPELICAN (18-input): DSP 1347, LUT 230,231, FF 63,343, latency 14cyc/0.070µs, II=1.

---

## 3. What's BLOCKED / IN PROGRESS

**GPU TensorFlow on the pod** (needed for a fast 1.21M-jet toptag retrain). The pod has an
**NVIDIA A10 (23GB), driver 595.71.05**. `libcuda.so.1` (driver) is fine at
`/usr/lib/x86_64-linux-gnu`.

**STORAGE FACT (learned 2026-07-01, the root cause of every failed install):** `/opt/conda`
is on the container overlay — wiped on every server reset, and writing GBs there trips the
k8s ephemeral-storage limit → pod eviction (the `[and-cuda]` install *completed* once, then
the pod was evicted and the env vanished). Only `/home/jovyan` (100G RBD) persists, including
`~/.cache/pip` (downloaded wheels survive resets) and `~/.condarc`. **The env must live at
`/home/jovyan/envs/fast_jetclass`** (see §4a). The pod base image ships TF 2.17/cu12 —
unusable here (qkeras 0.9 needs Keras-2-era TF 2.14).

---

## 4. RUNBOOK to the goal (do these on the pod)

### 4a. Create the GPU-TF env ON PERSISTENT STORAGE (survives resets)
`environment.yml` now pins `tensorflow[and-cuda]==2.14` directly (l1-jet-id 7d7a7d6).
```bash
cd ~/ClaudeInProgress/l1-jet-id && git pull origin toptag-comparison
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --append envs_dirs /home/jovyan/envs   # persists in ~/.condarc; enables activate-by-name
nohup conda env create -f environment.yml -p /home/jovyan/envs/fast_jetclass > ~/envcreate.log 2>&1 &
tail -f ~/envcreate.log        # wheels come from ~/.cache/pip (persistent) — minutes, not hours
conda activate /home/jovyan/envs/fast_jetclass
pip uninstall -y tensorrt      # ~1.6GB, pulled by [and-cuda], not needed by TF
# set CUDA + libstdc++ paths, verify GPU (put in ~/gpuenv.sh, re-source in every new shell):
NV=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$(echo $NV/*/lib | tr ' ' ':'):/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
**CPU fallback:** the DeepSet is tiny; a CPU run with `batch_size 1024`, `kfolds 2` on 1.21M
jets is a few hours — acceptable if GPU stays stuck.

### 4b. Retrain the DeepSet on toptag  ← the main deliverable
```bash
cd ~/ClaudeInProgress/l1-jet-id && git pull origin toptag-comparison
cd scripts
# IMPORTANT: --gpu 0 (NOT --gpu "" — empty string HIDES the GPU via CUDA_VISIBLE_DEVICES)
./deepsets_train --config configs/deepsets_8bit_20const_toptag/deepsets_8bit_20const_toptag.yml --gpu 0
./deepsets_test  --root_dir trained_deepsets/deepsets_8bit_20const_toptag --gpu 0 --seed 123 \
                 --h5_path ../../PELICAN-nano/data/toptag/test.h5
```
First log lines must say `GPU: NVIDIA A10` (not "No GPU detected"). Sanity: toptag AUC should be
sensible (>>0.5). The train-fit norm pkl lands at
`l1-jet-id/data/jetid_toptagset/processed/normparams_robust_20const_ptetaphi.pkl`.

### 4c. Regenerate the DeepSet equivariance overlay (the "new plots")
Two sub-decisions — see §6. Minimal path (DeepSet-only refresh, keeps existing nanoPELICAN
aggregates):
```bash
cd ~/ClaudeInProgress/nPELICAN-fpga && git pull origin deepset-equivariance
cd equivariance
conda activate fast_jetclass
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # matplotlib libstdc++
# DeepSet logits on the canonical boosted set, using the NEW toptag model + norm pkl:
python run_sweep_deepset.py \
  --l1-repo   /home/jovyan/ClaudeInProgress/l1-jet-id \
  --model-dir /home/jovyan/ClaudeInProgress/l1-jet-id/scripts/trained_deepsets/deepsets_8bit_20const_toptag/kfolding1 \
  --norm-pkl  /home/jovyan/ClaudeInProgress/l1-jet-id/data/jetid_toptagset/processed/normparams_robust_20const_ptetaphi.pkl \
  --nconst 20 --norm robust --mode both
python overlay_deepset.py --mode both      # -> results/plots_deepset_overlay/, aggregates_deepset.csv
```
**For a fully consistent toptag comparison** the canonical set + nanoPELICAN aggregates should
also be regenerated on `toptag/test.h5` — see §6.

### 4d. C6 ROC overlay (DeepSet vs nanoPELICAN on the same test set)
`roc_overlay.py` needs nanoPELICAN logits over the SAME test set. nanoPELICAN's are in
`nPELICAN-fpga/tb_data/full_pmu_test.dat` order (`full_signal.dat` labels). Easiest honest
comparison: run the DeepSet on those same 50k jets too. (Currently roc_overlay aligns to
test.h5 order — may need a small tweak to accept the toptag 50k order. Ask the user.)

---

## 5. What's STALE and needs refreshing (all from the old sample_data run)
- `comparison/{roc_summary.csv, REPORT.md}` DeepSet numbers (AUC 0.9518 etc.) — those are
  sample_data. Recompute on toptag after 4b.
- `equivariance/results/aggregates_*.csv` + `plots_*` — nanoPELICAN on the sample_data canonical
  set (AUC 0.70–0.89). If moving equivariance to toptag (§6), these regenerate.
- The old sample_data DeepSet checkpoint is being overwritten by 4b (fine).

---

## 6. OPEN DECISIONS (ask the user)
1. **Equivariance dataset.** For the equivariance overlay to be fully on toptag, set
   `equivariance/config.yaml: data_file: ../PELICAN-nano/data/toptag/test.h5`, then rerun
   `gen_boosted_inputs.py` (new canonical) + `run_sweep.py` (nanoPELICAN aggregates on toptag,
   needs torch/brevitas + g++) + `run_sweep_deepset.py` + `overlay_deepset.py`. NOTE: the
   equivariance *conclusion* (nanoPELICAN Lorentz-invariant, DeepSet drifts) is
   dataset-INDEPENDENT — only the absolute AUC panel changes. So a lighter option is to keep the
   equivariance on sample_data and just state the AUC caveat, OR move it to toptag for full
   consistency. Confirm which.
2. **C6 ROC alignment** on the toptag 50k vs test.h5 order (§4d).

---

## 7. Environment gotchas (all learned the hard way)
- `conda activate` needs `source "$(conda info --base)/etc/profile.d/conda.sh"` first (pod shell).
- **`--gpu 0`, never `--gpu ""`** — empty string sets CUDA_VISIBLE_DEVICES="" and hides the GPU.
- Env is `tensorflow[and-cuda]==2.14` + `qkeras==0.9` + `numpy==1.26.*` (TF 2.14 needs numpy<2).
  qkeras resolver can churn: `pip install qkeras==0.9 --no-deps` if it hangs.
- matplotlib CXXABI error → `conda install -c conda-forge 'libstdcxx-ng>=13'` +
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
- `pip install .` is non-editable → after `git pull`, either reinstall or use `pip install -e .`.
- JupyterHub pod gets culled on long downloads → run installs `nohup … &` from a **Terminal**
  (not a notebook cell); pip cache resumes.
- `trained_deepsets/` and `equivariance/{canonical/*.dat, results/logits_*.dat}` are gitignored
  (regenerable). Committed equivariance deliverables = `aggregates*.csv` + `plots*/*.png`.
- Vitis synth on pod: `export XILINX_{VIVADO,VITIS,HLS}=/tools/Xilinx/*/2023.2` + PATH; run
  `vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0"`.

## 8. Handy verified facts
- Dev Mac CAN run the firmware locally end-to-end: `PELICAN-nano/.venv/bin/python` has
  torch 2.12 + brevitas 0.12.1; `g++` + `build_local.sh` work; canonical set + all checkpoints
  are present. Used to prove the dataset finding (§0). run_sweep.py --config <one-model> works.
- `model_loader.py --max-input-bits N` is a Lever-2 CAP on input_t (only if N < bit-exact
  width); it does NOT widen dots (dot_t stays the learned input_quant grid). Do not expect it to
  change AUC materially.

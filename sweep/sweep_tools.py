#!/usr/bin/env python3
"""Helpers for the n-hidden capacity sweep (params vs accuracy vs FPGA resources).

Subcommands (each prints one line to stdout so the bash driver can capture it):

  count-params  --ckpt C --repo R      -> total learnable parameters (structural;
                                          rebuilds the model from the checkpoint's
                                          saved args, no state load needed)
  metrics       --prefix P --logdir L  -> accuracy,AUC,BgRej03  (last row of the
                                          trainer's <prefix>.Best.metrics.csv)
  set-nhidden   --header H --n N        -> rewrites `#define NHIDDEN N` in nPELICAN.h
  parse-csynth  --rpt F                 -> LUT,FF,DSP,BRAM,lat_cycles,lat_ns,II
                                          (top-level module row of a Vitis csynth rpt)
  parse-vsynth  --rpt F                 -> vLUT,vFF,vDSP,vBRAM  (Vivado post-synth
                                          report_utilization; the REAL resource counts)
  backfill      --csv C --n N --rpt F   -> fill HLS resource columns of the n_hidden==N
                                          row in an existing sweep_results.csv
  backfill-vsynth --csv C --n N --rpt F -> fill the vLUT/vFF/vDSP/vBRAM columns (adds
                                          them if the csv predates vsynth support)

Only count-params imports torch/Brevitas (and only under --quant checkpoints).
"""
import argparse
import os
import re
import sys


def count_params(ckpt_path, repo):
    import torch
    sys.path.insert(0, os.path.abspath(repo))
    from src.models import PELICANNano
    from src.layers.quant import QuantConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]

    qc = QuantConfig(
        enabled=a.quant,
        weight_bit_width=a.weight_bit_width,
        act_bit_width=a.act_bit_width,
        input_bit_width=a.input_bit_width,
        pmu_bit_width=a.pmu_bit_width,
        weight_per_channel=a.weight_per_channel,
        po2_scales=a.po2_scales,
        allow_alpha_scaling=a.allow_alpha_scaling,
    )
    # Mirror the construction in train_pelican_nano.py so the count is exact.
    model = PELICANNano(
        a.n_hidden,
        activate_agg=a.activate_agg, activate_lin=a.activate_lin,
        activation=a.activation, add_beams=a.add_beams,
        config=a.config, config_out=a.config_out, average_nobj=a.nobj_avg,
        factorize=a.factorize, masked=a.masked,
        activate_agg_out=a.activate_agg_out, activate_lin_out=a.activate_lin_out,
        scale=a.scale, dropout=a.dropout, drop_rate=a.drop_rate,
        drop_rate_out=a.drop_rate_out, batchnorm=a.batchnorm,
        quant_config=qc, device=torch.device("cpu"), dtype=torch.float,
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metrics(prefix, logdir):
    # trainer writes: <logdir>/<prefix.split('-')[0]>.Best.metrics.csv
    path = os.path.join(logdir, prefix.split("-")[0] + ".Best.metrics.csv")
    if not os.path.exists(path):
        return "NA,NA,NA"
    with open(path) as f:
        rows = [r.rstrip("\n") for r in f if r.strip()]
    if len(rows) < 2:
        return "NA,NA,NA"
    header = rows[0].split(",")
    last = rows[-1].split(",")
    row = dict(zip(header, last))
    acc = row.get("accuracy", "NA")
    auc = row.get("AUC", "NA")
    br = row.get("BgRejectionAt0.3", "NA")
    return f"{acc},{auc},{br}"


def set_nhidden(header, n):
    with open(header) as f:
        text = f.read()
    new, count = re.subn(
        r"(#define\s+NHIDDEN\s+)\d+", r"\g<1>%d" % n, text, count=1
    )
    if count != 1:
        sys.exit(f"ERROR: could not find `#define NHIDDEN` in {header}")
    with open(header, "w") as f:
        f.write(new)
    return f"NHIDDEN={n}"


def parse_csynth(rpt):
    # Header-aware parse of the "Synthesis Summary" hierarchy table. Column order and
    # spacing vary across Vitis versions, so we locate columns by their header names
    # (BRAM/DSP/FF/LUT/Interval/(cycles)/(ns)) and read the top-module row (first data
    # row whose name cell starts with '+'). The header spans two lines; the second one
    # carries the column names:
    #   |  & Loops | Type | Slack |(cycles)|(ns)| Latency|Interval|Count|Pipelined|BRAM|DSP|FF|LUT|URAM|
    def num(c):
        m = re.search(r"[\d.]+", c)
        return m.group(0) if m else "NA"

    with open(rpt) as f:
        lines = f.readlines()

    hdr = None
    hdr_i = None
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("|")]
        if "BRAM" in cells and "DSP" in cells and "LUT" in cells:
            hdr, hdr_i = cells, i
            break
    if hdr is None:
        return "NA,NA,NA,NA,NA,NA,NA"

    def col(name):
        return hdr.index(name) if name in hdr else None

    ci = {k: col(k) for k in ("(cycles)", "(ns)", "Interval", "BRAM", "DSP", "FF", "LUT")}
    for line in lines[hdr_i + 1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) <= max(v for v in ci.values() if v is not None):
            continue
        if len(cells) > 1 and cells[1].startswith("+"):  # top-module row (|+ name or | + name)
            def get(k):
                j = ci[k]
                return num(cells[j]) if j is not None and j < len(cells) else "NA"
            return "{},{},{},{},{},{},{}".format(
                get("LUT"), get("FF"), get("DSP"), get("BRAM"),
                get("(cycles)"), get("(ns)"), get("Interval"))
    return "NA,NA,NA,NA,NA,NA,NA"


def parse_vsynth(rpt):
    # Vivado report_utilization; grab the summary rows. Real counts (post-synth),
    # typically much lower than the HLS csynth estimate.
    labels = {
        "CLB LUTs": "vLUT", "CLB Registers": "vFF",
        "DSPs": "vDSP", "Block RAM Tile": "vBRAM",
    }
    out = {v: "NA" for v in labels.values()}
    with open(rpt) as f:
        for line in f:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 3:
                continue
            key = labels.get(cells[1].rstrip("*"))
            if key and out[key] == "NA":
                m = re.search(r"\d+", cells[2])
                if m:
                    out[key] = m.group(0)
    return "{vLUT},{vFF},{vDSP},{vBRAM}".format(**out)


def _backfill(csv_path, n, cols, vals, add_missing=False):
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    header = rows[0]
    for c in cols:
        if c not in header:
            if not add_missing:
                raise SystemExit(f"column {c} not in {csv_path}")
            header.append(c)
            for r in rows[1:]:
                r.append("NA")
    idx = {name: i for i, name in enumerate(header)}
    nh = idx["n_hidden"]
    hit = False
    for r in rows[1:]:
        if r and r[nh] == str(n):
            for name, val in zip(cols, vals):
                r[idx[name]] = val
            hit = True
    if not hit:
        return f"no row with n_hidden={n} in {csv_path}"
    with open(csv_path, "w", newline="") as f:
        _csv.writer(f).writerows(rows)
    return f"n_hidden={n}: {','.join(vals)}"


def backfill(csv_path, n, rpt):
    cols = ["LUT", "FF", "DSP", "BRAM", "lat_cycles", "lat_ns", "II"]
    return _backfill(csv_path, n, cols, parse_csynth(rpt).split(","))


def backfill_vsynth(csv_path, n, rpt):
    cols = ["vLUT", "vFF", "vDSP", "vBRAM"]
    return _backfill(csv_path, n, cols, parse_vsynth(rpt).split(","), add_missing=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("count-params"); s.add_argument("--ckpt", required=True); s.add_argument("--repo", required=True)
    s = sub.add_parser("metrics"); s.add_argument("--prefix", required=True); s.add_argument("--logdir", required=True)
    s = sub.add_parser("set-nhidden"); s.add_argument("--header", required=True); s.add_argument("--n", type=int, required=True)
    s = sub.add_parser("parse-csynth"); s.add_argument("--rpt", required=True)
    s = sub.add_parser("parse-vsynth"); s.add_argument("--rpt", required=True)
    s = sub.add_parser("backfill"); s.add_argument("--csv", required=True); s.add_argument("--n", required=True); s.add_argument("--rpt", required=True)
    s = sub.add_parser("backfill-vsynth"); s.add_argument("--csv", required=True); s.add_argument("--n", required=True); s.add_argument("--rpt", required=True)

    a = p.parse_args()
    if a.cmd == "count-params":
        print(count_params(a.ckpt, a.repo))
    elif a.cmd == "metrics":
        print(metrics(a.prefix, a.logdir))
    elif a.cmd == "set-nhidden":
        print(set_nhidden(a.header, a.n))
    elif a.cmd == "parse-csynth":
        print(parse_csynth(a.rpt))
    elif a.cmd == "parse-vsynth":
        print(parse_vsynth(a.rpt))
    elif a.cmd == "backfill":
        print(backfill(a.csv, a.n, a.rpt))
    elif a.cmd == "backfill-vsynth":
        print(backfill_vsynth(a.csv, a.n, a.rpt))


if __name__ == "__main__":
    main()

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
  backfill      --csv C --n N --rpt F   -> fill the resource columns of the n_hidden==N
                                          row in an existing sweep_results.csv

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
    # Top-level row looks like:
    # |+ nPELICAN | Timing| -0.02| 14| 70.000| -| 1| -| yes| -| 1347 (10%)| 63343 (1%)| 229779 (13%)| -|
    with open(rpt) as f:
        for line in f:
            if re.match(r"\s*\|\+\s", line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                # cells: name,issue,slack,lat_cyc,lat_ns,iter,interval,trip,
                #        pipelined,BRAM,DSP,FF,LUT,URAM
                def num(c):
                    m = re.search(r"[\d.]+", c)
                    return m.group(0) if m else "NA"
                lat_cyc = num(cells[3])
                lat_ns = num(cells[4])
                ii = num(cells[6])
                bram = num(cells[9])
                dsp = num(cells[10])
                ff = num(cells[11])
                lut = num(cells[12])
                return f"{lut},{ff},{dsp},{bram},{lat_cyc},{lat_ns},{ii}"
    return "NA,NA,NA,NA,NA,NA,NA"


def backfill(csv_path, n, rpt):
    import csv as _csv
    res = parse_csynth(rpt).split(",")  # LUT,FF,DSP,BRAM,lat_cycles,lat_ns,II
    cols = ["LUT", "FF", "DSP", "BRAM", "lat_cycles", "lat_ns", "II"]
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    nh = idx["n_hidden"]
    hit = False
    for r in rows[1:]:
        if r and r[nh] == str(n):
            for name, val in zip(cols, res):
                r[idx[name]] = val
            hit = True
    if not hit:
        return f"no row with n_hidden={n} in {csv_path}"
    with open(csv_path, "w", newline="") as f:
        _csv.writer(f).writerows(rows)
    return f"n_hidden={n}: {','.join(res)}"


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("count-params"); s.add_argument("--ckpt", required=True); s.add_argument("--repo", required=True)
    s = sub.add_parser("metrics"); s.add_argument("--prefix", required=True); s.add_argument("--logdir", required=True)
    s = sub.add_parser("set-nhidden"); s.add_argument("--header", required=True); s.add_argument("--n", type=int, required=True)
    s = sub.add_parser("parse-csynth"); s.add_argument("--rpt", required=True)
    s = sub.add_parser("backfill"); s.add_argument("--csv", required=True); s.add_argument("--n", required=True); s.add_argument("--rpt", required=True)

    a = p.parse_args()
    if a.cmd == "count-params":
        print(count_params(a.ckpt, a.repo))
    elif a.cmd == "metrics":
        print(metrics(a.prefix, a.logdir))
    elif a.cmd == "set-nhidden":
        print(set_nhidden(a.header, a.n))
    elif a.cmd == "parse-csynth":
        print(parse_csynth(a.rpt))
    elif a.cmd == "backfill":
        print(backfill(a.csv, a.n, a.rpt))


if __name__ == "__main__":
    main()

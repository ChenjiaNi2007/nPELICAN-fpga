"""
model_loader.py  — export a nanoPELICAN checkpoint to firmware/weights/weights.h

Merged version:
  * Auto-detects pre-refactor (.coefs) vs post-refactor (.mixing.weight) checkpoints.
  * Infers NHIDDEN / NOUT from tensor shapes.
  * Float path runs STANDALONE (stub-unpickling, no PELICAN-nano install needed).
  * Quant path (--quant) rebuilds the model through Brevitas and exports the
    SNAPPED on-grid weights via quant_weight().value — the authoritative grid,
    including --po2-scales rounding. It does NOT re-implement quantization by
    hand (a hand-rolled absmax/127 snap at the wrong bit width silently
    corrupts QAT weights).
  * --split-types emits w1_t / w2_t array types (requires those typedefs in
    nPELICAN.h); default emits weight_t for drop-in compatibility.

Usage
-----
  Float checkpoint:
    python3 model_loader.py --model path/to/best.pt

  QAT checkpoint (bit widths default from the checkpoint's saved args):
    python3 model_loader.py --model path/to/qat_best.pt --quant \
        --repo ../PELICAN-nano [--split-types]
"""
import sys
import os
import math
import types
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Export nanoPELICAN checkpoint to weights.h')
parser.add_argument('--model', nargs=1, required=True,
                    help='Path to PELICAN-nano checkpoint .pt file')
parser.add_argument('--quant', action='store_true', default=False,
                    help='Checkpoint was trained with --quant; extract snapped on-grid weights via Brevitas')
parser.add_argument('--repo', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'PELICAN-nano'),
                    help='Path to the PELICAN-nano repo root (needed for --quant)')
parser.add_argument('--n-hidden', type=int, default=None,
                    help='Override inferred NHIDDEN')
parser.add_argument('--weight-bit-width', type=int, default=None,
                    help='Override bit width (default: read from checkpoint args)')
parser.add_argument('--act-bit-width', type=int, default=None)
parser.add_argument('--input-bit-width', type=int, default=None)
parser.add_argument('--no-po2', action='store_true',
                    help='Set if trained WITHOUT --po2-scales')
parser.add_argument('--split-types', action='store_true', default=False,
                    help='Emit w1_t / w2_t array types instead of weight_t (typedefs must exist in nPELICAN.h)')
parser.add_argument('--out', type=str, default='weights/weights.h')
parser.add_argument('--out-types', type=str, default=None,
                    help='Path for generated typedef header (default: dirname(--out)/types_generated.h). '
                         'The firmware build expects both weights.h and types_generated.h under firmware/weights/.')
args = parser.parse_args()

if args.out_types is None:
    args.out_types = os.path.join(os.path.dirname(args.out) or '.', 'types_generated.h')

np.set_printoptions(precision=15, floatmode='fixed')
torch.set_printoptions(precision=15)

# ---------------------------------------------------------------------------
# Import strategy:
#   --quant: we MUST import the real PELICAN-nano package (and brevitas) to
#            rebuild the model. Try that first.
#   float:   stub modules suffice for unpickling; no install needed.
# ---------------------------------------------------------------------------
_real_import = False
if args.quant:
    sys.path.insert(0, os.path.abspath(args.repo))
    try:
        import logging
        logging.disable(logging.CRITICAL)
        from src.layers.quant import QuantConfig          # noqa: F401
        from src.models.pelican_nano import PELICANNano   # noqa: F401
        _real_import = True
    except ImportError as e:
        sys.exit(f'ERROR: --quant requires the PELICAN-nano repo and brevitas.\n'
                 f'  Tried repo path: {os.path.abspath(args.repo)}\n'
                 f'  Import error: {e}\n'
                 f'  Pass the correct path with --repo.')

if not _real_import:
    # Register stubs so torch.load can unpickle scheduler/args objects
    # without the full src/ package installed.
    def _make_stub_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        def _stub_getattr(attr_name: str) -> type:
            return type(attr_name, (), {
                '__init__': lambda self, *a, **k: None,
                '__setstate__': lambda self, d: self.__dict__.update(d),
            })
        mod.__getattr__ = _stub_getattr
        return mod

    for _modname in [
        'src', 'src.trainer', 'src.trainer.scheduler',
        'src.trainer.args', 'src.trainer.utils', 'src.trainer.trainer',
    ]:
        if _modname not in sys.modules:
            sys.modules[_modname] = _make_stub_module(_modname)

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
m = torch.load(args.model[0], map_location='cpu', weights_only=False)
sd = m['model_state']
margs = m.get('args', None)

def _arg(name, fallback=None):
    return getattr(margs, name, fallback) if margs is not None else fallback

# ---------------------------------------------------------------------------
# Detect key format
# ---------------------------------------------------------------------------
_has_new = 'net2to2.eq_layers.0.mixing.weight' in sd
_has_old = 'net2to2.eq_layers.0.coefs' in sd
if not _has_new and not _has_old:
    sys.exit('ERROR: checkpoint contains neither "net2to2.eq_layers.0.coefs" nor '
             '"net2to2.eq_layers.0.mixing.weight". Is this a nanoPELICAN checkpoint?')
fmt = 'new' if _has_new else 'old'
print(f'Checkpoint format: {"post-refactor (mixing.weight)" if fmt == "new" else "pre-refactor (coefs)"}')

# ---------------------------------------------------------------------------
# Infer NHIDDEN / NOUT from shapes
# ---------------------------------------------------------------------------
if fmt == 'new':
    NHIDDEN = sd['net2to2.eq_layers.0.mixing.weight'].shape[0]   # [NHIDDEN, 6]
    NOUT    = sd['agg_2to0.mixing.weight'].shape[0]              # [NOUT, NHIDDEN*2]
else:
    NHIDDEN = sd['net2to2.eq_layers.0.coefs'].shape[1]           # [1, NHIDDEN, 6]
    NOUT    = sd['agg_2to0.coefs'].shape[1]                      # [NHIDDEN, NOUT, 2]
if args.n_hidden is not None and args.n_hidden != NHIDDEN:
    print(f'WARNING: --n-hidden {args.n_hidden} overrides inferred NHIDDEN={NHIDDEN}')
    NHIDDEN = args.n_hidden
print(f'NHIDDEN={NHIDDEN}, NOUT={NOUT}')

# ---------------------------------------------------------------------------
# C initialiser formatting
# ---------------------------------------------------------------------------
def _c(arr):
    return (np.array2string(np.asarray(arr), separator=', ')
            .replace('\n', '').replace('[', '{').replace(']', '}'))

# ---------------------------------------------------------------------------
# Batchnorm (keys unchanged between formats; float during QAT by design)
# ---------------------------------------------------------------------------
mean1   = sd['net2to2.message_layers.0.normlayer.running_mean'].item()
weight1 = sd['net2to2.message_layers.0.normlayer.weight'].item()
var1    = sd['net2to2.message_layers.0.normlayer.running_var'].item()
bias1   = sd['net2to2.message_layers.0.normlayer.bias'].item()
batch1  = np.array((mean1, weight1 / np.sqrt(var1), bias1))

mean2   = sd['msg_2to0.normlayer.running_mean'].numpy()
weight2 = sd['msg_2to0.normlayer.weight'].numpy()
var2    = sd['msg_2to0.normlayer.running_var'].numpy()
bias2   = sd['msg_2to0.normlayer.bias'].numpy()
batch2  = np.column_stack((mean2, weight2 / np.sqrt(var2), bias2))

# ---------------------------------------------------------------------------
# Float path — raw state-dict tensors
# ---------------------------------------------------------------------------
def _extract_float_weights():
    if fmt == 'new':
        w1  = np.ravel(sd['net2to2.eq_layers.0.mixing.weight'].numpy())
        b2  = sd['agg_2to0.mixing.bias'].numpy()
        w2  = np.ravel(sd['agg_2to0.mixing.weight'].numpy())
    else:
        w1  = np.ravel(sd['net2to2.eq_layers.0.coefs'].numpy()[0])
        w2  = np.ravel(sd['agg_2to0.coefs'].numpy())
        b2  = sd['agg_2to0.bias'].numpy()[0]
    b1  = sd['net2to2.eq_layers.0.bias'].numpy()
    b1d = sd['net2to2.eq_layers.0.diag_bias'].numpy()
    return w1, b1, b1d, w2, b2

# ---------------------------------------------------------------------------
# Quant path — rebuild the model and read snapped weights through Brevitas.
# This is the authoritative grid (correct bit width AND po2 rounding); do NOT
# replace this with a hand-rolled absmax snap.
# ---------------------------------------------------------------------------
# Populated by _extract_quant_weights() so the typedef generator can read the
# learned per-quantizer scales/signedness off the SAME rebuilt Brevitas model.
_quant_info = {}


def _po2_k(scale, who):
    """k = -log2(scale); hard error if the learned scale is not an exact po2."""
    if scale <= 0:
        sys.exit(f'ERROR: {who} has non-positive scale {scale!r}; cannot derive a fixed-point type.')
    k = -math.log2(scale)
    kr = round(k)
    if abs(k - kr) > 1e-6:
        sys.exit(f'ERROR: {who} scale {scale:.9e} is not a power of two '
                 f'(k = -log2(scale) = {k:.6f}, not integer). '
                 f'Was the checkpoint trained with --po2-scales?')
    return int(kr)


def _read_act_quant(model, qnn, calibrated=False):
    """Read learned scale/signedness/bit-width for each QuantIdentity/QuantReLU
    via named_modules(). If any act scale is uninitialized, run ONE training-mode
    calibration forward on data/sample_data before load_state_dict (handled by the
    caller); for the target 24-bit checkpoint the scale buffers already exist."""
    act = {}
    act_types = (qnn.QuantIdentity, qnn.QuantReLU)
    for name, module in model.named_modules():
        if not isinstance(module, act_types):
            continue
        scale = module.act_quant.scale()
        if scale is None:
            return None  # signal: needs calibration
        s = float(scale.reshape(-1)[0])
        signed = bool(module.act_quant.is_signed)
        bw = int(round(float(module.act_quant.bit_width())))
        act[name] = dict(scale=s, signed=signed, bits=bw, module=type(module).__name__)
    return act


def _calibration_forward(model, repo):
    """ONE training-mode forward on data/sample_data to populate act scales
    (CLAUDE.md gotcha: model.train() -> forward -> [caller does load_state_dict]).
    Only invoked when an act scale is uninitialized in the rebuilt model."""
    sys.path.insert(0, os.path.abspath(repo))
    from src.dataloaders.collate import collate_fn  # noqa: F401 (best-effort)
    import h5py  # noqa: F401
    # Minimal smoke batch from sample_data; mirrors check_scales.py's note that a
    # single training-mode forward initializes scaling_impl.value.
    sample = os.path.join(os.path.abspath(repo), 'data', 'sample_data', 'valid.h5')
    raise RuntimeError(
        'Act-quantizer scales were uninitialized and a calibration forward is required. '
        f'Implement the sample-data forward on {sample}. '
        '(For the target 24-bit checkpoint the scale buffers already exist, so this path '
        'is not exercised; see model_loader.py:_calibration_forward.)')


def _extract_quant_weights():
    if fmt == 'old':
        sys.exit('ERROR: --quant requires a post-refactor checkpoint (mixing.weight keys). '
                 'Convert it first with scripts/convert_checkpoint.py.')

    import brevitas.nn as qnn

    wbw = args.weight_bit_width or _arg('weight_bit_width', 24)
    abw = args.act_bit_width    or _arg('act_bit_width', 24)
    ibw = args.input_bit_width  or _arg('input_bit_width', 24)
    po2 = (not args.no_po2) if margs is None else _arg('po2_scales', not args.no_po2)

    qcfg = QuantConfig(enabled=True, weight_bit_width=wbw, act_bit_width=abw,
                       input_bit_width=ibw, po2_scales=po2)

    def _build():
        return PELICANNano(NHIDDEN, quant_config=qcfg,
                           batchnorm=_arg('batchnorm', 'b'),
                           activation=_arg('activation', 'relu'))

    model = _build()
    model.load_state_dict(sd)
    model.eval()

    # Act-quantizer scales (and signedness) off the rebuilt model.
    act_info = _read_act_quant(model, qnn)
    if act_info is None:
        # Uninitialized act scales: calibrate on sample_data, THEN reload (strict).
        model = _build()
        model.train()
        _calibration_forward(model, args.repo)
        model.load_state_dict(sd)
        model.eval()
        act_info = _read_act_quant(model, qnn)
        if act_info is None:
            sys.exit('ERROR: act-quantizer scales still uninitialized after calibration.')

    print(f'\nQuant config: weight/act/input bits = {wbw}/{abw}/{ibw}, po2={po2}')
    print('QuantLinear weight scales:')
    qw1 = model.net2to2.eq_layers[0].mixing.quant_weight()
    qw2 = model.agg_2to0.mixing.quant_weight()
    weight_info = {}
    for name, qw in (('net2to2.eq_layers.0', qw1), ('agg_2to0', qw2)):
        s = float(qw.scale.reshape(-1)[0])
        signed = bool(qw.signed) if hasattr(qw, 'signed') and qw.signed is not None else True
        bw = int(round(float(qw.bit_width)))
        weight_info[name] = dict(scale=s, signed=signed, bits=bw)
        print(f'  {name:<22} scale = {s:.6e}  ~ 2^{math.log2(s):.2f}'
              f'  => {-math.log2(s):.0f} fractional bits')

    print('Activation / identity quantizer scales:')
    for name, info in act_info.items():
        s = info['scale']
        print(f'  {name:<34} scale = {s:.6e}  ~ 2^{math.log2(s):.2f}'
              f'  => {-math.log2(s):.0f} frac bits  signed={info["signed"]}')

    _quant_info['act'] = act_info
    _quant_info['weight'] = weight_info

    w1  = np.ravel(qw1.value.detach().numpy())
    w2  = np.ravel(qw2.value.detach().numpy())
    b1  = sd['net2to2.eq_layers.0.bias'].numpy()
    b1d = sd['net2to2.eq_layers.0.diag_bias'].numpy()
    b2  = sd['agg_2to0.mixing.bias'].numpy()
    return w1, b1, b1d, w2, b2

# ---------------------------------------------------------------------------
# Generated typedef header (--quant only). Per-quantizer fixed-point types are
# derived from the learned Brevitas scales (k = -log2(scale)); accumulator/MAC
# types are derived by formula from those + term counts. NOTHING is hardcoded to
# 24 bits — bit widths come from the quantizers, so retraining at 16/16/16
# regenerates correctly with zero manual header edits.
# ---------------------------------------------------------------------------
def _emit_types_header(path, act_info, weight_info, b1, b1d, b2):
    # --- map module names -> typedef names + provenance label ---
    # Quantization-point typedefs, one per quantizer.
    act = act_info
    wgt = weight_info

    def _pt(scale, signed, bits, who):
        """(W, I, signed) for a quantization-point type at learned scale 2^-k."""
        k = _po2_k(scale, who)
        return bits, bits - k, signed, k

    dot_W, dot_I, dot_s, dot_k       = _pt(act['input_quant']['scale'], act['input_quant']['signed'], act['input_quant']['bits'], 'input_quant')
    t2_W, t2_I, t2_s, t2_k           = _pt(act['net2to2.eq_layers.0.post_agg_quant']['scale'], act['net2to2.eq_layers.0.post_agg_quant']['signed'], act['net2to2.eq_layers.0.post_agg_quant']['bits'], 'post_agg 2->2')
    relu_W, relu_I, relu_s, relu_k   = _pt(act['net2to2.eq_layers.0.act_layer']['scale'], act['net2to2.eq_layers.0.act_layer']['signed'], act['net2to2.eq_layers.0.act_layer']['bits'], 'act_layer (ReLU)')
    t0_W, t0_I, t0_s, t0_k           = _pt(act['agg_2to0.post_agg_quant']['scale'], act['agg_2to0.post_agg_quant']['signed'], act['agg_2to0.post_agg_quant']['bits'], 'post_agg 2->0')
    out_W, out_I, out_s, out_k       = _pt(act['output_quant']['scale'], act['output_quant']['signed'], act['output_quant']['bits'], 'output_quant')
    w1_W, w1_I, w1_s, w1_k           = _pt(wgt['net2to2.eq_layers.0']['scale'], wgt['net2to2.eq_layers.0']['signed'], wgt['net2to2.eq_layers.0']['bits'], '2->2 weights')
    w2_W, w2_I, w2_s, w2_k           = _pt(wgt['agg_2to0']['scale'], wgt['agg_2to0']['signed'], wgt['agg_2to0']['bits'], '2->0 weights')

    # All quantization-point types share one bit width per design; use the input
    # quantizer's bit width as B for the formula-derived accumulator/MAC types.
    B = dot_W

    # --- bias_t_gen / bn_t_gen with the documented range asserts ---
    bias_max = float(max(abs(np.asarray(b1)).max(), abs(np.asarray(b1d)).max(), abs(np.asarray(b2)).max()))
    if not (bias_max < 8):
        sys.exit(f'ERROR: max|bias| = {bias_max} >= 8; bias_t_gen (I=4) cannot represent it.')
    bn_max = float(np.abs(np.asarray(batch1)).max())
    bn_max = max(bn_max, float(np.abs(np.asarray(batch2)).max()))
    if not (bn_max < 2 ** 8):
        sys.exit(f'ERROR: max|BN const| = {bn_max} >= 2^8; bn_t_gen (I=9) cannot represent it.')

    # --- accumulator headroom from NPARTICLES2 (NPARTICLES + 2 spurions).
    # NPARTICLES2 is a firmware constant the loader already mirrors (NPELICAN.h:
    # NPARTICLES2 = NPARTICLES + 2 = 22); accumulators are NOT covered by any
    # learned scale, so they get explicit integer headroom over the summand type.
    NPARTICLES = 20
    NPARTICLES2 = NPARTICLES + 2
    H2 = math.ceil(math.log2(NPARTICLES2 ** 2))   # full-sum headroom = 9
    H1 = math.ceil(math.log2(NPARTICLES2))        # row-sum headroom  = 5

    acc2_W, acc2_I       = B + H2, t2_I + H2
    accrow_W, accrow_I   = B + H1, t2_I + H1
    acc0_W, acc0_I       = B + H2, t0_I + H2
    acc0row_W, acc0row_I = B + H1, t0_I + H1

    # --- MAC temporaries: EXACT product width + term-count headroom ---
    # The product weight*operand is exact at F = F(weight)+F(operand) fractional bits; keep
    # all of them so the dense sum carries no internal rounding before the relu/out quantizer
    # (PyTorch does this MAC in float, which is ~exact relative to those 2^-22 / 2^-23 grids).
    # I = I(weight)+I(operand)+ceil(log2(#terms)) gives integer headroom for the sum.
    w1_F, t2_F = w1_W - w1_I, t2_W - t2_I
    w2_F, t0_F = w2_W - w2_I, t0_W - t0_I
    mac2_terms = 6 + 1 + 1            # 6 products w1*t2 + bias + diag_bias = 8 summands
    mac0_terms = 2 * NHIDDEN + 1      # 2*NHIDDEN products w2*t0 + bias
    mac2_I = w1_I + t2_I + math.ceil(math.log2(mac2_terms))
    mac2_W = mac2_I + (w1_F + t2_F)
    mac0_I = w2_I + t0_I + math.ceil(math.log2(mac0_terms))
    mac0_W = mac0_I + (w2_F + t0_F)

    def _fixed(W, I, signed, rnd='AP_RND_CONV', sat='AP_SAT'):
        base = 'ap_fixed' if signed else 'ap_ufixed'
        return f'{base}<{W}, {I}, {rnd}, {sat}>'

    L = []
    L.append('#ifndef NPELICAN_TYPES_GENERATED_H_')
    L.append('#define NPELICAN_TYPES_GENERATED_H_')
    L.append('')
    L.append('// GENERATED by model_loader.py --quant. Do not hand-edit.')
    L.append('// Per-quantizer fixed-point types derived from the checkpoint\'s learned')
    L.append('// Brevitas scales (k = -log2(scale); I = bits - k). Accumulator / MAC types')
    L.append('// derived by formula from those + term counts. Phase 1: INERT include — not')
    L.append('// yet wired into the datapath (Phase 2 swaps usage and retires the old types).')
    L.append('')
    L.append('#include "ap_fixed.h"')
    L.append('')
    L.append('#define NPELICAN_GENERATED_TYPES 1')
    L.append('')
    L.append('// ---- Quantization-point types: ap_fixed<B, B-k, AP_RND_CONV, AP_SAT> ----')

    def _pt_line(tname, W, I, signed, scale, k, who):
        sg = 'signed' if signed else 'unsigned'
        return (f'typedef {_fixed(W, I, signed)} {tname};'
                f'  // {who} ({sg}): scale=2^-{k} ({scale:.9e}), bits={W}, k={k}')

    L.append(_pt_line('dot_t',    dot_W,  dot_I,  dot_s,  act['input_quant']['scale'], dot_k, 'input_quant'))
    L.append(_pt_line('t2_t',     t2_W,   t2_I,   t2_s,   act['net2to2.eq_layers.0.post_agg_quant']['scale'], t2_k, 'post_agg 2->2'))
    L.append(_pt_line('relu_t',   relu_W, relu_I, relu_s, act['net2to2.eq_layers.0.act_layer']['scale'], relu_k, 'act_layer (QuantReLU)'))
    L.append(_pt_line('t0_t',     t0_W,   t0_I,   t0_s,   act['agg_2to0.post_agg_quant']['scale'], t0_k, 'post_agg 2->0'))
    L.append(_pt_line('out_t',    out_W,  out_I,  out_s,  act['output_quant']['scale'], out_k, 'output_quant'))
    L.append(_pt_line('w1_gen_t', w1_W,   w1_I,   w1_s,   wgt['net2to2.eq_layers.0']['scale'], w1_k, '2->2 weights'))
    L.append(_pt_line('w2_gen_t', w2_W,   w2_I,   w2_s,   wgt['agg_2to0']['scale'], w2_k, '2->0 weights'))
    L.append('')
    L.append('// ---- Float-trained biases / BatchNorm constants / normalization constants ----')
    L.append('// These are NOT PyTorch quantization points (PyTorch keeps them in float), so')
    L.append('// per CLAUDE.md/plan they are WIDENED, not snapped: their fixed-point rounding')
    L.append('// error must stay below half the LSB of the next real quantizer they feed.')
    # bias_t_gen: feeds the dense MAC then the relu/out quantizers (2^-22 / 2^-23). A 20-frac
    # type (B=24,I=4) rounds some learned biases to ~2^-22, exceeding half the relu LSB; widen
    # to F=24 frac (W=28) so |err| <= 2^-25.
    BIAS_W, BIAS_I = 28, 4
    L.append(f'typedef ap_fixed<{BIAS_W}, {BIAS_I}, AP_RND_CONV, AP_SAT> bias_t_gen;'
             f'  // b1,b1_diag,b2 (float); |bias|max={bias_max:.6g}<8 (I={BIAS_I}); F={BIAS_W-BIAS_I}, err<=2^-{BIAS_W-BIAS_I+1}')
    # bn_t_gen: the BN scale gamma/sigma multiplies (dots-mean), |dots-mean| up to ~10^3, so it
    # needs |err| < 2^-29 (F>=29) to keep batch1 within half the t2 LSB (2^-19); the BN beta adds
    # straight into batch1 needing F>=21. mean ~O(100) needs I>=9. Use <40,9> (F=31).
    BN_W, BN_I = 40, 9
    L.append(f'typedef ap_fixed<{BN_W}, {BN_I}, AP_RND_CONV, AP_SAT> bn_t_gen;'
             f'  // BN mean/scale/beta (float); |c|max={bn_max:.6g}<2^{BN_I-1} (I={BN_I}); F={BN_W-BN_I}')
    # norm_t: invnave=1/N̄, invnave2=1/N̄^2 (not po2). A 12-frac internal_t mis-rounds invnave2
    # by ~40%. They multiply the raw aggregation sums feeding 2^-18/2^-23 grids; F=39 gives
    # |err|~2^-33, safe even times the widest accumulator (~2^15).
    NORM_W, NORM_I = 40, 1
    L.append(f'typedef ap_fixed<{NORM_W}, {NORM_I}, AP_RND_CONV, AP_SAT> norm_t;'
             f'  // 1/N̄, 1/N̄^2 normalize-late multipliers (F={NORM_W-NORM_I})')
    L.append('')
    L.append('// ---- Accumulators (normalize-late: raw sum first, ONE rescale after; see')
    L.append('//      CLAUDE.md). Default ap_fixed rounding/overflow ON PURPOSE — accumulation')
    L.append('//      must be EXACT, so no AP_RND_CONV here; integer headroom = ceil(log2(#terms))')
    L.append(f'//      over the summand type. NPARTICLES2 = NPARTICLES+2 = {NPARTICLES2}.')
    L.append(f'//      H2 = ceil(log2(NPARTICLES2^2)) = {H2}; H1 = ceil(log2(NPARTICLES2)) = {H1}.')
    L.append(f'typedef ap_fixed<{acc2_W}, {acc2_I}> acc2_t;     // jmass raw sum of t2-range summands (B+H2, I(t2_t)+H2)')
    L.append(f'typedef ap_fixed<{accrow_W}, {accrow_I}> accrow_t;   // jdotp row sums (B+H1, I(t2_t)+H1)')
    L.append(f'typedef ap_fixed<{acc0_W}, {acc0_I}> acc0_t;     // R full sum (B+H2, I(t0_t)+H2)')
    L.append(f'typedef ap_fixed<{acc0row_W}, {acc0row_I}> acc0row_t;  // trace (B+H1, I(t0_t)+H1)')
    L.append('')
    L.append('// ---- MAC temporaries: I = I(weight)+I(operand)+ceil(log2(#terms)), W = I+B ----')
    L.append(f'typedef ap_fixed<{mac2_W}, {mac2_I}> mac2_t;     // 2->2 dense: 6 w1*t2 products + b1 + b1_diag = {mac2_terms} terms')
    L.append(f'typedef ap_fixed<{mac0_W}, {mac0_I}> mac0_t;     // 2->0 dense: 2*NHIDDEN w2*t0 products + b2 = {mac0_terms} terms')
    L.append('')
    L.append('#endif  // NPELICAN_TYPES_GENERATED_H_')
    L.append('')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(L))
    print(f'Wrote {path}')

    # Return scale comment lines to append to weights.h (D3: scales on the record).
    cl = ['', '//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----']
    for nm, info in act.items():
        kk = _po2_k(info['scale'], nm)
        cl.append(f'//  {nm:<34} scale=2^-{kk} ({info["scale"]:.9e}) signed={info["signed"]} bits={info["bits"]}')
    for nm, info in wgt.items():
        kk = _po2_k(info['scale'], nm)
        cl.append(f'//  {nm+" (weights)":<34} scale=2^-{kk} ({info["scale"]:.9e}) signed={info["signed"]} bits={info["bits"]}')
    return cl


# ---------------------------------------------------------------------------
# Dispatch + write weights.h (format unchanged apart from optional types)
# ---------------------------------------------------------------------------
if args.quant:
    print('Extracting snapped dequantized weights (QAT path, via Brevitas)...')
    w1_2to2, b1_2to2, b1d_2to2, w2_2to0, b2_2to0 = _extract_quant_weights()
else:
    w1_2to2, b1_2to2, b1d_2to2, w2_2to0, b2_2to0 = _extract_float_weights()

# Element typedefs for the weights.h arrays. Under --quant the firmware datapath
# (nPELICAN.cpp, Phase 2) uses the GENERATED per-stage types, so the arrays are declared
# with those; the float path keeps the original hand-written names. Array names, sizes,
# element order and VALUES are frozen either way — only the element typedef name changes.
if args.quant:
    w1_type, w2_type = 'w1_gen_t', 'w2_gen_t'
    bn_type, bias_type, norm_type = 'bn_t_gen', 'bias_t_gen', 'norm_t'
else:
    w1_type = 'w1_t' if args.split_types else 'weight_t'
    w2_type = 'w2_t' if args.split_types else 'weight_t'
    bn_type, bias_type, norm_type = 'weight_t', 'bias_t', 'internal_t'

# Emit the generated typedef header (--quant only) and collect the scale comment
# lines to append to weights.h for the record (plan D3).
_scale_comment_lines = []
if args.quant:
    _scale_comment_lines = _emit_types_header(
        args.out_types, _quant_info['act'], _quant_info['weight'],
        b1_2to2, b1d_2to2, b2_2to0)

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w') as f:
    f.write('#include "../nPELICAN.h"\n')
    f.write('//model: ' + str(_arg('prefix', '?')) + '\n')
    f.write('//nobj: ' + str(_arg('nobj', '?')) + '\n\n')

    nobj_avg = _arg('nobj_avg', 49)
    f.write('//normalization constants\n')
    f.write('//nobj avg = {}\n'.format(nobj_avg))
    f.write(f'{norm_type} invnave = {1 / nobj_avg};\n')
    f.write(f'{norm_type} invnave2 = {1 / nobj_avg ** 2};\n\n')

    f.write('//first batchnorm [mean, weight/sqrt(var), bias]\n')
    f.write(f'{bn_type} batch1_2to2[3] = ' + _c(batch1) + ';\n\n')

    f.write('//2to2 linear layer\n')
    f.write(f'{w1_type} w1_2to2[NHIDDEN*6] = ' + _c(w1_2to2) + ';\n')
    f.write(f'{bias_type} b1_2to2[NHIDDEN] = ' + _c(b1_2to2) + ';\n')
    f.write(f'{bias_type} b1_diag_2to2[NHIDDEN] = ' + _c(b1d_2to2) + ';\n\n')

    f.write('//second batchnorm [channel][mean, weight/sqrt(var), bias]\n')
    f.write(f'{bn_type} batch2_2to0[NHIDDEN][3] = ' + _c(batch2) + ';\n\n')

    f.write('//2to1 linear layer\n')
    f.write(f'{w2_type} w2_2to0[NHIDDEN*2*NOUT] = ' + _c(w2_2to0) + ';\n')
    f.write(f'{bias_type} b2_2to0[NOUT] = ' + _c(b2_2to0) + ';\n')

    # D3: measured QAT scales appended as comments for the record (quant path only).
    if _scale_comment_lines:
        f.write('\n'.join(_scale_comment_lines) + '\n')

print(f'\nWrote {args.out}  (NHIDDEN={NHIDDEN}, NOUT={NOUT})')
"""
model_loader.py  — export a nanoPELICAN checkpoint to firmware/weights/weights.h

Merged version:
  * Auto-detects pre-refactor (.coefs) vs post-refactor (.mixing.weight) checkpoints.
  * Infers NHIDDEN / NOUT from tensor shapes.
  * Jet spurion (ckpt args.add_jet): NSPURIONS=3 -> NPELICAN_JET_SPURION (jet_input port).
  * --jet-quant-split (ckpt args.jet_quant_split): input_quant_jet / input_quant_mjet /
    pmu_quant_jet -> NPELICAN_JET_QUANT_SPLIT + dotj_t / dotm_t / jet_t (+ dotall_t, the
    exact container of all three dot grids). Without it they alias dot_t / input_t.
  * Nonlinear head (head.weight present, ckpt args.head_hidden=K): agg_2to0 mixes to K
    channels -> QuantReLU (relu0_t) -> head QuantLinear(K, NOUT) -> NPELICAN_HEAD K,
    N2TO0_OUT K, w_head/b_head. --quant only (the float path refuses jet/head ckpts).
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
parser.add_argument('--pmu-bit-width', type=int, default=None,
                    help='Override momentum-quantizer (pmu_quant) bit width for checkpoints '
                         'trained with --pmu-bit-width (default: read from checkpoint args). '
                         'When present, input_t is the TRAINED momentum grid and both the '
                         'analytic width bound and --max-input-bits are bypassed.')
parser.add_argument('--input-unsigned', action=argparse.BooleanOptionalAction, default=None,
                    help='Override the d_ij grid signedness (default: read from the '
                         'checkpoint args). Trained with --input-unsigned, dot_t is '
                         'ap_ufixed and one bit moves from sign to magnitude. Brevitas '
                         'derives scale() from the SAME stored stat divided by a '
                         'signedness-dependent threshold, so rebuilding with the wrong '
                         'signedness loads cleanly but silently yields a 2x-off scale.')
parser.add_argument('--no-po2', action='store_true',
                    help='Set if trained WITHOUT --po2-scales')
parser.add_argument('--split-types', action='store_true', default=False,
                    help='Emit w1_t / w2_t array types instead of weight_t (typedefs must exist in nPELICAN.h)')
parser.add_argument('--out', type=str, default='weights/weights.h')
parser.add_argument('--bn-frac-bits', type=int, default=None,
                    help='Cap the fractional width of bn_t_gen (default: derived as '
                         't2_F + dot_mag + 2). The BN1 scale is a scalar constant '
                         'multiplying every dot, so its literal width binds every BN1 '
                         'multiplier at once. Measured on xcu250 @5ns: an 8-bit literal '
                         'lands in fabric (~49 LUT each), a 10-bit literal takes a DSP48 '
                         'per multiply AND violates timing; 9 bits is untested. In Bind Op '
                         'reports the module operand is the literal width PLUS one '
                         'zero-extension bit (8->mul_6s_9ns, 10->mul_6s_11ns). Check the '
                         'printed literal width before trusting a resource number.')
parser.add_argument('--bn-eps', type=float, default=1e-5,
                    help='BatchNorm eps used in the scale weight/sqrt(var+eps); must match '
                         'the training MaskedBatchNorm eps (default 1e-5).')
parser.add_argument('--out-types', type=str, default=None,
                    help='Path for generated typedef header (default: dirname(--out)/types_generated.h). '
                         'The firmware build expects both weights.h and types_generated.h under firmware/weights/.')
parser.add_argument('--max-input-bits', type=int, default=None,
                    help='Cap input_t total width to N bits (Lever 2): trades dot4 front-end '
                         'precision for DSP. The dots are only dot_t-bit; the un-capped width '
                         'buys bit-exact agreement with PyTorch near rounding boundaries. '
                         'Capping below that shrinks the dot multiplier (e.g. 18 -> mul_18s_18s '
                         '= 1 DSP instead of 2). NOT free: re-check the online golden gate after '
                         'lowering it. Only INPUT_F is reduced; INPUT_I (range) is preserved so '
                         'momenta never saturate. Caps <= INPUT_I give NEGATIVE F (ap_fixed<W,I> '
                         'with I > W, momentum LSB 2^-F > 1 GeV) — legal, but expect heavy dot4 '
                         'precision loss. No effect if N >= the derived bit-exact width.')
parser.add_argument('--pmu-extra-int-bits', type=int, default=0,
                    help='INFERENCE-ONLY ablation for trained-pmu checkpoints: add K integer bits '
                         'to input_t, doubling the momentum clip K times (e.g. 1: 512 -> 1024 GeV). '
                         'By default W grows by K too (same momentum LSB, so un-clipped jets stay '
                         'bit-identical to the trained grid); with --pmu-keep-width W is held and '
                         'the LSB coarsens by 2^K. Either way the firmware no longer matches the '
                         'PyTorch pmu_quant on clipped momenta — not a deployable export.')
parser.add_argument('--pmu-keep-width', action='store_true', default=False,
                    help='With --pmu-extra-int-bits: hold input_t width fixed (coarser LSB).')
parser.add_argument('--bias-guard-bits', type=int, default=8,
                    help='Extra fractional bits kept in the bias literals beyond the MAC product '
                         'grid, so the float bias is not snapped onto that grid and cannot create '
                         'exact rounding ties at the following quantizer; 0 reproduces the '
                         'pre-2026-10-06 single-grid behaviour.')
parser.add_argument('--agg-guard-bits', type=int, default=8,
                    help='Extra fractional bits for the stored BatchNorm1 output (bn1out_t) and its '
                         'raw-sum accumulators beyond max(t2_F, t0_F)+1, so the N2-term aggregation sums '
                         'cannot tip a post-aggregation rounding tie after the 1/Nbar rescale. 0 = the '
                         'pre-2026-10-06 rule (default: 8).')
parser.add_argument('--norm-guard-bits', type=int, default=8,
                    help='Extra fractional bits for the 1/Nbar, 1/Nbar^2 normalize-late literals (norm_t) beyond '
                         'the exact-product rule, so their non-po2 rounding error cannot flip a post-aggregation '
                         'rounding tie (default: 8; 0 = pre-2026-10-07 rule).')
parser.add_argument('--bn-guard-bits', type=int, default=0,
                    help='Extra fractional bits for the BatchNorm literals (bn_t_gen) beyond the derived '
                         't2_F + dot_mag + 2. Default 0 keeps the DSP-tuned literal width; 8 removes the '
                         'last post-agg tie flips on jet-spurion checkpoints at DSP/timing cost. --bn-frac-bits '
                         'still caps the result.')
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
    N2TO0_OUT = sd['agg_2to0.mixing.weight'].shape[0]            # [K or NOUT, NHIDDEN*2]
    if 'head.weight' in sd:
        # --head-hidden K: 2->0 mixes down to K hidden channels, head maps K -> NOUT
        NOUT = sd['head.weight'].shape[0]                         # [NOUT, K]
        HEAD = N2TO0_OUT
        if sd['head.weight'].shape[1] != HEAD:
            sys.exit(f'ERROR: head.weight is {tuple(sd["head.weight"].shape)} but '
                     f'agg_2to0.mixing has {HEAD} output rows.')
    else:
        NOUT = N2TO0_OUT
        HEAD = 0
else:
    NHIDDEN = sd['net2to2.eq_layers.0.coefs'].shape[1]           # [1, NHIDDEN, 6]
    NOUT    = sd['agg_2to0.coefs'].shape[1]                      # [NHIDDEN, NOUT, 2]
    N2TO0_OUT, HEAD = NOUT, 0
# Spurions prepended before the constituents: 2 beams (+ the full-jet 4-vector when the
# checkpoint was trained with --add-jet). Mirrors PELICAN-nano collate_fn(add_jet).
NSPURIONS = 3 if bool(_arg('add_jet', False)) else 2
# --jet-quant-split: three learned d_ij quantizers (pairs / jet row+col / m_jet^2) and a
# separate jet-momentum quantizer. Model-shaping: the rebuild must replay it.
JET_QUANT_SPLIT = bool(_arg('jet_quant_split', False))
if not JET_QUANT_SPLIT and any(k.split('.')[0] in ('input_quant_jet', 'input_quant_mjet', 'pmu_quant_jet')
                               for k in sd):
    sys.exit('ERROR: checkpoint has input_quant_jet/input_quant_mjet/pmu_quant_jet keys but its '
             'args do not record jet_quant_split=True. Refusing to guess the dot grids.')
if JET_QUANT_SPLIT and NSPURIONS != 3:
    sys.exit('ERROR: checkpoint records jet_quant_split=True without add_jet (the split '
             'quantizes the jet spurion row/col at slot 2).')
if args.n_hidden is not None and args.n_hidden != NHIDDEN:
    print(f'WARNING: --n-hidden {args.n_hidden} overrides inferred NHIDDEN={NHIDDEN}')
    NHIDDEN = args.n_hidden
print(f'NHIDDEN={NHIDDEN}, NOUT={NOUT}, head_hidden={HEAD}, NSPURIONS={NSPURIONS}, '
      f'jet_quant_split={JET_QUANT_SPLIT}')

# ---------------------------------------------------------------------------
# C initialiser formatting
# ---------------------------------------------------------------------------
def _c(arr):
    return (np.array2string(np.asarray(arr), separator=', ')
            .replace('\n', '').replace('[', '{').replace(']', '}'))

# ---------------------------------------------------------------------------
# Batchnorm (keys unchanged between formats; float during QAT by design).
# The firmware applies (x-mean)*scale+bias with a precomputed scale; that scale MUST
# include the BN eps exactly as PyTorch does: scale = weight / sqrt(var + eps). Omitting
# eps silently inflates the scale for small-variance channels (here BN2 ch0 var~1.7e-4,
# eps=1e-5 -> a 2.8% scale error), which breaks bit-exactness downstream.
# ---------------------------------------------------------------------------
BN_EPS = args.bn_eps   # MaskedBatchNorm default is 1e-5 (src/layers/masked_batchnorm.py)

mean1   = sd['net2to2.message_layers.0.normlayer.running_mean'].item()
weight1 = sd['net2to2.message_layers.0.normlayer.weight'].item()
var1    = sd['net2to2.message_layers.0.normlayer.running_var'].item()
bias1   = sd['net2to2.message_layers.0.normlayer.bias'].item()
batch1  = np.array((mean1, weight1 / np.sqrt(var1 + BN_EPS), bias1))

mean2   = sd['msg_2to0.normlayer.running_mean'].numpy()
weight2 = sd['msg_2to0.normlayer.weight'].numpy()
var2    = sd['msg_2to0.normlayer.running_var'].numpy()
bias2   = sd['msg_2to0.normlayer.bias'].numpy()
batch2  = np.column_stack((mean2, weight2 / np.sqrt(var2 + BN_EPS), bias2))

# ---------------------------------------------------------------------------
# Float path — raw state-dict tensors
# ---------------------------------------------------------------------------
def _extract_float_weights():
    if HEAD > 0 or NSPURIONS != 2:
        # The jet/head firmware switches (NPELICAN_JET_SPURION, NPELICAN_HEAD, N2TO0_OUT,
        # NSPURIONS) live in types_generated.h, which only the --quant path writes; weights.h
        # is included too late to define them. Refuse rather than emit a header the firmware
        # would silently compile as a no-jet / no-head model.
        sys.exit(f'ERROR: checkpoint has head_hidden={HEAD} / add_jet={NSPURIONS == 3}; the '
                 f'float (no --quant) export does not support the jet spurion or the '
                 f'nonlinear head. Export a QAT checkpoint with --quant.')
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

# Guard bits below the finest block-FP mantissa LSB carried by the input_t PORT.
# See the derivation at the block-FP branch of _emit_types_header for why 3 (the
# analytic minimum) is not enough and 8 is. Costs wires, never DSP.
NPELICAN_BFP_GUARD_BITS = 8


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
    pbw = args.pmu_bit_width if args.pmu_bit_width is not None else _arg('pmu_bit_width', None)
    if pbw is None and any(k.startswith('pmu_quant.') for k in sd):
        sys.exit('ERROR: checkpoint has pmu_quant keys (trained momentum quantizer) but no '
                 'recorded bit width; pass --pmu-bit-width matching the training flag.')

    # --- Lever 7: per-particle block floating point on the momenta ---------------
    # BlockFPQuant is STATELESS (no params, no buffers), so it leaves no trace in the
    # state_dict and _read_act_quant() cannot see it — it is neither a QuantIdentity
    # nor a QuantReLU. The checkpoint's own `args` are therefore the ONLY record of
    # the momentum grid, and they must be replayed both into the rebuilt model and
    # into the emitted types. Same class of silent failure as input_unsigned /
    # input_clip_min: a rebuild that drops the flag load_state_dict's cleanly and then
    # describes a grid the model never used.
    bfp = bool(_arg('pmu_block_fp', False))
    bfp_emin = int(_arg('pmu_exp_min', 0))
    bfp_emax = int(_arg('pmu_exp_max', 10))
    if bfp:
        if pbw is None:
            sys.exit('ERROR: checkpoint records --pmu-block-fp but no pmu_bit_width; the '
                     'mantissa width is not recoverable. Pass --pmu-bit-width.')
        if pbw < 3:
            sys.exit(f'ERROR: block-FP mantissa width {pbw} < 3 (needs sign + I=2).')
        if bfp_emin > bfp_emax:
            sys.exit(f'ERROR: pmu_exp_min ({bfp_emin}) > pmu_exp_max ({bfp_emax}).')
        # The firmware encoder realigns with a RIGHT shift only (p >> e). A negative
        # exponent would need a left shift and extra integer headroom in mraw_t — not
        # implemented, and never used (momenta are in GeV, the trained clamp is [0,10]).
        # Refuse rather than emit a header the firmware would silently mis-scale.
        if bfp_emin < 0:
            sys.exit(f'ERROR: pmu_exp_min={bfp_emin} < 0 is not supported by the firmware '
                     f'block-FP encoder (right-shift realign only). See '
                     f'firmware/np_blockfp.h.')
        if 'pmu_block_fp' not in QuantConfig.__dataclass_fields__:
            sys.exit('ERROR: checkpoint was trained with --pmu-block-fp but the '
                     'PELICAN-nano checkout at --repo has no such QuantConfig field. '
                     'Update it, or the rebuilt model uses a uniform momentum grid.')

    # --- SPS: static per-slot learned exponent (block-FP with a TRAINED constant
    # exponent per particle slot instead of the runtime LZC). Unlike dynamic block-FP
    # it DOES leave state (pmu_quant.log2_exp / pmu_quant.exp_initialized), but the
    # rebuild still has to be told the mode and the slot count, or load_state_dict
    # fails (or, worse, a dynamic rebuild of a static run describes the wrong grid).
    # Slot layout = firmware P1Prep: slots 0,1 beams, 2.. constituents, so
    # n_slots = nobj + 2 must equal NPARTICLES2.
    bfp_static = bool(_arg('pmu_static_exp', False))
    bfp_nslots = None
    if 'pmu_quant.log2_exp' in sd and not bfp_static:
        sys.exit('ERROR: checkpoint has pmu_quant.log2_exp (static per-slot exponent) but '
                 'its args do not record pmu_static_exp=True. Refusing to guess the grid.')
    if bfp_static:
        if not bfp:
            sys.exit('ERROR: checkpoint records pmu_static_exp=True without pmu_block_fp.')
        for _f in ('pmu_static_exp', 'pmu_n_slots'):
            if _f not in QuantConfig.__dataclass_fields__:
                sys.exit(f'ERROR: checkpoint was trained with --pmu-static-exp but the '
                         f'PELICAN-nano checkout at --repo has no QuantConfig field {_f!r}. '
                         f'Update it, or the rebuilt model uses a dynamic exponent.')
        if not bool(_arg('add_beams', True)):
            sys.exit('ERROR: static per-slot exponent checkpoint trained without beams '
                     '(add_beams=False); the firmware slot layout (P1Prep) always '
                     'prepends the 2 beam spurions.')
        _nobj = _arg('nobj', None)
        _nobj = 20 if _nobj is None else int(_nobj)
        bfp_nslots = _nobj + NSPURIONS

    # Signedness of the d_ij grid. MUST match training: Brevitas derives scale() from the
    # same stored stat divided by a signedness-dependent int threshold (2^(b-1)-1 signed
    # vs 2^b-1 unsigned), so a wrong rebuild still load_state_dict's cleanly and then
    # reports a scale that is off by ~2x — a silent bit-exactness break. Verified below.
    # The checkpoint's own record is authoritative; checkpoints predating the flag have
    # no such arg and were signed, which the False fallback gives. An explicit override
    # that contradicts the record is refused — it is exactly how you get a header that
    # describes a grid the model never trained on.
    ck_iuns = bool(_arg('input_unsigned', False))
    iuns = ck_iuns if args.input_unsigned is None else bool(args.input_unsigned)
    if margs is not None and args.input_unsigned is not None and iuns != ck_iuns:
        sys.exit(f'ERROR: --{"" if iuns else "no-"}input-unsigned contradicts the '
                 f'checkpoint, which records input_unsigned={ck_iuns}. dot_t and every '
                 f'type derived from it would describe a grid the model never used. '
                 f'Omit the flag to follow the checkpoint.')

    qkw = dict(enabled=True, weight_bit_width=wbw, act_bit_width=abw,
               input_bit_width=ibw, po2_scales=po2)
    if pbw is not None:
        # only when set, so the loader still works against an older PELICAN-nano
        # checkout whose QuantConfig has no pmu_bit_width field
        qkw['pmu_bit_width'] = pbw
    if bfp:
        # Replayed so the rebuilt model runs the SAME momentum grid it trained on.
        # (Field presence was asserted above.) BlockFPQuant is stateless, so this
        # changes no stored scale — but a rebuild that silently used a uniform
        # QuantIdentity here would make every calibration/verification forward wrong.
        qkw['pmu_block_fp'] = True
        qkw['pmu_exp_min'] = bfp_emin
        qkw['pmu_exp_max'] = bfp_emax
        if bfp_static:
            qkw['pmu_static_exp'] = True
            qkw['pmu_n_slots'] = bfp_nslots
            # SPS exponent-floor / fixed-exponent training options. The static module
            # always carries their buffers and exponent_table() already applies the
            # floor, but replay them anyway (rebuild-trap rule) when the checkout knows
            # the fields; a checkpoint that USED them against a checkout that lacks them
            # is refused rather than silently rebuilt without.
            for _f, _v in (('pmu_exp_floor_batches', int(_arg('pmu_exp_floor_batches', 0) or 0)),
                           ('pmu_exp_fixed', bool(_arg('pmu_exp_fixed', False)))):
                if _f in QuantConfig.__dataclass_fields__:
                    qkw[_f] = _v
                elif _v:
                    sys.exit(f'ERROR: checkpoint was trained with {_f}={_v} but the '
                             f'PELICAN-nano checkout at --repo has no such QuantConfig '
                             f'field. Update it.')
    if iuns:
        if 'input_unsigned' not in QuantConfig.__dataclass_fields__:
            sys.exit('ERROR: checkpoint needs input_unsigned but the PELICAN-nano checkout '
                     f'at --repo has no such QuantConfig field. Update it.')
        qkw['input_unsigned'] = True

    # input_clip_min floors the d_ij saturation point. It is NOT a training-only
    # constraint: Brevitas stores the raw runtime stat and applies the clamp on every
    # forward, so a rebuild that omits it reports the unclamped scale and every derived
    # type is wrong — same silent failure as the signedness above. Always replay it.
    icm = _arg('input_clip_min', None)
    if icm is not None:
        if 'input_clip_min' not in QuantConfig.__dataclass_fields__:
            sys.exit('ERROR: checkpoint was trained with --input-clip-min but the '
                     'PELICAN-nano checkout at --repo has no such QuantConfig field. '
                     'Update it, or the emitted dot_t will use an unclamped scale.')
        qkw['input_clip_min'] = float(icm)
    if JET_QUANT_SPLIT:
        # Replayed (model-shaping: adds input_quant_jet / input_quant_mjet / pmu_quant_jet).
        if 'jet_quant_split' not in QuantConfig.__dataclass_fields__:
            sys.exit('ERROR: checkpoint was trained with --jet-quant-split but the PELICAN-nano '
                     'checkout at --repo has no such QuantConfig field. Update it.')
        if bfp:
            sys.exit('ERROR: jet_quant_split with pmu_block_fp is not supported (PyTorch refuses it).')
        qkw['jet_quant_split'] = True
        # Per-quantizer widths (--jet-input-bit-width / --mjet-input-bit-width / --jet-pmu-bit-width).
        # MUST be replayed: the rebuilt module's bit_width() is what sizes dotj_t/dotm_t/jet_t, and
        # load_state_dict succeeds even with the wrong width (only the clip stat is stored), so a
        # missing replay silently emits 6/6/12-bit jet types for a 10/16/20-bit checkpoint
        # (caught by the real-checkpoint gate: 51/200).
        for _f in ('jet_input_bit_width', 'mjet_input_bit_width', 'jet_pmu_bit_width'):
            _v = _arg(_f, None)
            if _v is not None:
                if _f not in QuantConfig.__dataclass_fields__:
                    sys.exit(f'ERROR: checkpoint sets {_f}={_v} but the PELICAN-nano checkout at --repo '
                             f'has no such QuantConfig field; update it.')
                qkw[_f] = int(_v)
    qcfg = QuantConfig(**qkw)

    def _build():
        # n_out only for K-class heads: older PELICAN-nano checkouts' constructor
        # lacks the kwarg, and NOUT=1 is their (only) default.
        extra = {'n_out': NOUT} if NOUT != 1 else {}
        if HEAD > 0:
            extra['head_hidden'] = HEAD
        return PELICANNano(NHIDDEN, quant_config=qcfg,
                           batchnorm=_arg('batchnorm', 'b'),
                           activation=_arg('activation', 'relu'), **extra)

    model = _build()
    model.load_state_dict(sd)
    model.eval()
    if JET_QUANT_SPLIT:
        _jq = [getattr(model, n, None) for n in ('input_quant_jet', 'input_quant_mjet')]
        if any(q is None for q in _jq) or not bool(getattr(model, 'jet_quant_split', False)):
            sys.exit('ERROR: checkpoint records jet_quant_split=True but the rebuilt model has '
                     'no input_quant_jet / input_quant_mjet.')
        if (pbw is not None) != (getattr(model, 'pmu_quant_jet', None) is not None):
            sys.exit('ERROR: rebuilt pmu_quant_jet presence disagrees with pmu_bit_width.')
        for _n, _q in zip(('input_quant_jet', 'input_quant_mjet'), _jq):
            if bool(_q.act_quant.is_signed) == iuns:
                sys.exit(f'ERROR: rebuilt {_n} signedness disagrees with input_unsigned={iuns}.')

    # Same guard for the momentum grid: the rebuilt pmu_quant must actually BE the
    # block-FP module, with the mantissa width and exponent clamp the run recorded.
    # BlockFPQuant leaves nothing in the state_dict, so this assertion is the only
    # thing standing between a dropped flag and a header describing the wrong grid.
    if bfp:
        pq = getattr(model, 'pmu_quant', None)
        if pq is None or type(pq).__name__ != 'BlockFPQuant':
            sys.exit(f'ERROR: checkpoint records --pmu-block-fp but the rebuilt model\'s '
                     f'pmu_quant is {type(pq).__name__ if pq is not None else "None"}, not '
                     f'BlockFPQuant. The emitted mantissa/exponent types would describe a '
                     f'grid the model never used.')
        if (int(pq.bit_width), int(pq.exp_min), int(pq.exp_max)) != (pbw, bfp_emin, bfp_emax):
            sys.exit(f'ERROR: rebuilt BlockFPQuant is W={pq.bit_width} exp=[{pq.exp_min},'
                     f'{pq.exp_max}] but the checkpoint records W={pbw} exp=[{bfp_emin},'
                     f'{bfp_emax}].')
        _quant_info['blockfp'] = dict(bits=int(pq.bit_width), exp_min=int(pq.exp_min),
                                      exp_max=int(pq.exp_max),
                                      from_energy=bool(pq.from_energy))
        if bfp_static:
            # Firmware NPARTICLES2 = NPARTICLES + 2 beams; NPARTICLES is emitted into
            # types_generated.h from the checkpoint's nobj (fallback 20 -> 22).
            _FW_NPARTICLES2 = ((int(_arg('nobj')) if _arg('nobj', None) is not None else 20)
                               + NSPURIONS)
            if not bool(getattr(pq, 'static', False)):
                sys.exit('ERROR: checkpoint records pmu_static_exp=True but the rebuilt '
                         'BlockFPQuant is not static.')
            if not bool(pq.exp_initialized):
                sys.exit('ERROR: static BlockFPQuant exp_initialized=False after '
                         'load_state_dict: the exponent table was never trained/initialized.')
            etab = [int(v) for v in pq.exponent_table().reshape(-1).tolist()]
            if len(etab) != _FW_NPARTICLES2:
                sys.exit(f'ERROR: static exponent table has {len(etab)} slots but the '
                         f'firmware has NPARTICLES2={_FW_NPARTICLES2} (beams + NPARTICLES). '
                         f'Checkpoint nobj={bfp_nslots - NSPURIONS}; the firmware NPARTICLES '
                         f'(from checkpoint args.nobj, else 20) must equal it.')
            bad = [v for v in etab if v < max(bfp_emin, 0) or v > bfp_emax]
            if bad:
                sys.exit(f'ERROR: static exponent table entries {bad} outside '
                         f'[{bfp_emin},{bfp_emax}] (or negative; firmware right-shifts only).')
            _quant_info['blockfp']['exp_table'] = etab

    # Guard the silent failure described above: the rebuilt input quantizer must have the
    # signedness we asked for. A mismatch here means every dot_t-derived type is wrong.
    got_signed = bool(model.input_quant.act_quant.is_signed)
    if got_signed == iuns:
        sys.exit(f'ERROR: rebuilt input_quant is signed={got_signed} but input_unsigned='
                 f'{iuns} was requested. dot_t and every type derived from it would be '
                 f'wrong. Pass --input-unsigned/--no-input-unsigned to match training.')

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

    print(f'\nQuant config: weight/act/input bits = {wbw}/{abw}/{ibw}, '
          f'pmu={pbw if pbw is not None else "off"}, po2={po2}')
    print('QuantLinear weight scales:')
    qw1 = model.net2to2.eq_layers[0].mixing.quant_weight()
    qw2 = model.agg_2to0.mixing.quant_weight()
    weight_info = {}
    _wq = [('net2to2.eq_layers.0', qw1), ('agg_2to0', qw2)]
    if HEAD > 0:
        if getattr(model, 'head', None) is None:
            sys.exit('ERROR: checkpoint has head.* keys but the rebuilt model has no head '
                     '(PELICAN-nano checkout at --repo too old for --head-hidden?).')
        qwh = model.head.quant_weight()
        _wq.append(('head', qwh))
        if 'agg_2to0.act_layer' not in [n for n, _ in model.named_modules()]:
            sys.exit('ERROR: head checkpoint but the rebuilt agg_2to0 has no act_layer.')
    for name, qw in _wq:
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
    if HEAD > 0:
        # head weights row-major (NOUT, K) -> w_head[o*K + k]; float bias (D6)
        _quant_info['head'] = dict(w=np.ravel(qwh.value.detach().numpy()),
                                   b=sd['head.bias'].numpy())
    return w1, b1, b1d, w2, b2

# ---------------------------------------------------------------------------
# Generated typedef header (--quant only). Per-quantizer fixed-point types are
# derived from the learned Brevitas scales (k = -log2(scale)); accumulator/MAC
# types are derived by formula from those + term counts. NOTHING is hardcoded to
# 24 bits — bit widths come from the quantizers, so retraining at 16/16/16
# regenerates correctly with zero manual header edits.
# ---------------------------------------------------------------------------
def _momentum_absmax(repo, default=2048.0):
    """|p|max over the sample 4-momenta, used to size input_t's INTEGER width.
    This is a physics quantity (the momentum dynamic range) and is independent of
    the QAT bit-width flags. Falls back to `default` (the historical 2048 GeV
    assumption -> I=12) if the sample data is unavailable, so the loader never
    fails just because the dataset isn't checked out."""
    # The checkpoint's OWN dataset first (args.datadir, relative to the repo root or
    # absolute): e.g. the 5-class hls4ml files reach |p| ~2.5 TeV, so sizing from the
    # toptag sample would clip them. Then the toptag sample, then `default`.
    cands = []
    ddir = _arg('datadir', None)
    if ddir:
        ddir = ddir if os.path.isabs(ddir) else os.path.join(os.path.abspath(repo), ddir)
        cands += [os.path.join(ddir, 'valid.h5'), os.path.join(ddir, 'test.h5')]
    cands.append(os.path.join(os.path.abspath(repo), 'data', 'sample_data', 'valid.h5'))
    try:
        import h5py
    except Exception as e:  # noqa: BLE001
        print(f'  (input_t: h5py unavailable [{e}]); using |p|max default {default}')
        return float(default)
    errs = []
    for path in cands:
        try:
            with h5py.File(path, 'r') as f:
                pmax = float(np.abs(f['Pmu'][:]).max())
                if NSPURIONS == 3:
                    # --add-jet: the full-jet 4-vector (~1-5 TeV) also enters input_t
                    jmax = float(np.abs(f['Pjet'][:]).max())
                    print(f'  input_t: |Pjet|max = {jmax:.6g} (jet spurion) from {path}')
                    pmax = max(pmax, jmax)
            print(f'  input_t: |p|max = {pmax:.6g} from {path}')
            return pmax
        except Exception as e:  # noqa: BLE001 - any I/O / key error -> next candidate
            errs.append(f'{path}: {e}')
    print(f'  (input_t: 4-momenta unavailable [{"; ".join(errs)}]); '
          f'using |p|max default {default}')
    return float(default)


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

    # --- bias_t_gen / bn_t_gen widths DERIVED from the actual constant magnitudes ---
    # (not hardcoded — the int width must grow with the checkpoint, e.g. BN running_mean
    # scales with jet energy and can exceed a few hundred). _int_bits returns a signed
    # integer-bit count I whose ap_fixed range +/-2^(I-1) strictly exceeds the magnitude.
    def _int_bits(maxabs, floor_bits=2):
        if maxabs <= 0:
            return floor_bits
        return max(floor_bits, int(math.floor(math.log2(maxabs))) + 2)

    t2_F = t2_W - t2_I                       # post_agg-2to2 grid fractional bits
    relu_F = relu_W - relu_I                  # act_layer (ReLU) grid fractional bits
    out_F = out_W - out_I                     # output_quant grid fractional bits
    dot_F = dot_W - dot_I                      # input_quant (dots) grid fractional bits
    # Magnitude bits of the dot range: ap_fixed<W,I> spans +-2^(I-1), ap_ufixed<W,I>
    # spans [0, 2^I). Downstream sizing (BN_F, bn1out_t) bounds |dots|, so it must
    # follow the signedness or an unsigned dot_t under-sizes them by a full bit.
    dot_mag = dot_I - 1 if dot_s else dot_I

    # --- --jet-quant-split: the Gram matrix carries THREE learned grids (pairs: dot_t;
    # jet row/col: dotj_t; m_jet^2: dotm_t). The dots array holds all of them exactly in
    # dotall_t (I = max I, F = max F). Every range/precision bound that previously used the
    # dot_t span (input_t F, BN_F, bn1out_t) now uses dotall_t's. Without the split dotall
    # == dot, so these stay byte-identical.
    if JET_QUANT_SPLIT:
        _aj, _am = act['input_quant_jet'], act['input_quant_mjet']
        dotj_W, dotj_I, dotj_s, dotj_k = _pt(_aj['scale'], _aj['signed'], _aj['bits'], 'input_quant_jet')
        dotm_W, dotm_I, dotm_s, dotm_k = _pt(_am['scale'], _am['signed'], _am['bits'], 'input_quant_mjet')
        if not (dot_s == dotj_s == dotm_s):
            sys.exit('ERROR: input_quant / input_quant_jet / input_quant_mjet signedness differ.')
        dotall_I = max(dot_I, dotj_I, dotm_I)
        dotall_F = max(dot_F, dotj_W - dotj_I, dotm_W - dotm_I)
        dotall_W = dotall_I + dotall_F
        print(f'  jet quant split: dot_t k={dot_k} (I={dot_I}), dotj_t k={dotj_k} (I={dotj_I}), '
              f'dotm_t k={dotm_k} (I={dotm_I}) -> dotall_t <{dotall_W},{dotall_I}>')
    else:
        dotall_I, dotall_F, dotall_W = dot_I, dot_F, dot_W
    dotall_mag = dotall_I - 1 if dot_s else dotall_I

    # --- input_t: raw-momentum / IO type feeding dot4 (the 36x36 multipliers that
    # dominate DSP). NOT a learned quantizer (input_quant grids the DOTS, dot_t), so it
    # is widened analytically. Two independent parts:
    #   I = _int_bits(|p|max): physics dynamic range of the momenta (flag-independent).
    #   F: dot4 sums 4 products p1[k]*p2[k]; each operand rounds at 2^-(F+1), so the dot
    #      error is <= 4*|p|max*2^-F. For the firmware dot to round onto the SAME dot_t
    #      grid point as PyTorch's float dot we need that < 1/2 dot_t LSB = 2^-(dot_F+1):
    #        4*|p|max*2^-F <= 2^-(dot_F+1)  =>  F >= ceil(log2|p|max) + dot_F + 3.
    #      F therefore tracks the dot_t grid: coarser dots (lower QAT bits) -> smaller F
    #      -> narrower input_t -> each 36x36 dot multiply shrinks (4 DSP -> fewer/1 DSP).
    pmax = _momentum_absmax(args.repo)
    pmu = act.get('pmu_quant')
    bfp = _quant_info.get('blockfp')
    _pj = None   # pmu_quant_jet (only under --jet-quant-split with a trained pmu grid)
    if bfp is not None:
        # --- Lever 7: PER-PARTICLE BLOCK FLOATING POINT ---------------------------
        # p_k = m_k * 2^e with ONE exponent shared by a particle's four components:
        #   e   = clamp(floor(log2|E|), EXP_MIN, EXP_MAX)   -- one LZC on the energy
        #   m_k = p_k * 2^-e  on a signed W-bit, I=2 grid   -- LSB 2^-(W-2)
        # I=2 is provably sufficient: for a physical particle E >= |p_k| and
        # 2^e <= E < 2^(e+1), so |m_k| < 2 always (blockfp.py "Representation").
        #
        # The momentum grid is therefore RELATIVE, not absolute, and no single
        # input_t can describe it — hence the mantissa/exponent type pair below.
        # input_t survives as the raw-momentum PORT type feeding the on-chip
        # encoder (firmware/np_blockfp.h); it is not a multiplier operand any more,
        # so its width costs wires, not DSP. The dot4 multipliers are W x W.
        if not bfp.get('from_energy', True):
            sys.exit('ERROR: checkpoint uses BlockFPQuant(from_energy=False) (4-way max '
                     'over |p_k|); firmware/np_blockfp.h implements the E-based exponent '
                     'only. They are measured identical, but the header must not claim a '
                     'grid the firmware does not build.')
        MANT_W = int(bfp['bits'])
        MANT_I = 2
        MANT_F = MANT_W - MANT_I
        EXP_MIN = int(bfp['exp_min'])
        EXP_MAX = int(bfp['exp_max'])
        # Unsigned exponent field (EXP_MIN >= 0 is enforced at rebuild time).
        EXP_BITS = max(1, int(EXP_MAX).bit_length())
        SHIFT_BITS = max(1, int(2 * EXP_MAX).bit_length())

        INPUT_I = _int_bits(pmax)
        # The encoder rounds p*2^-e onto the mantissa grid, but p has ALREADY been
        # rounded into input_t at the port. For that double rounding to land where
        # PyTorch's single rounding of the float momenta lands, input_t's LSB must sit
        # well under the FINEST mantissa LSB, which occurs at the smallest exponent:
        #   mantissa LSB(e) = 2^(e-(W-2))  ->  finest at e = EXP_MIN
        #   need 2^-(F+1) <= (1/4) * 2^(EXP_MIN-(W-2))  =>  F >= (W-2) - EXP_MIN + 3.
        #
        # 3 guard bits is the ANALYTIC minimum and it is not enough in practice. Two
        # boundary effects survive it, both driven by the port rounding rather than by
        # the mantissa grid:
        #   (a) EXPONENT boundaries. e is decided from |E| AFTER it lands in input_t, so
        #       a float energy just under a power of two (1023.999 at F=8 -> 1024.0)
        #       picks e one higher than PyTorch did and lands the particle on a
        #       different mantissa grid entirely.
        #   (b) MANTISSA ties. PyTorch rounds the float momentum once; the firmware
        #       rounds float -> input_t -> mant_t, and a value near a mant_t half-grid
        #       point can be pushed across by the first rounding.
        # Both shrink by 2x per guard bit. NPELICAN_BFP_GUARD_BITS was raised to 8 after
        # measuring the golden gate on a W=7 checkpoint: at 3 guard bits the block-FP
        # front end contributed 7 mismatching events / 500 over the dots-injected
        # baseline; at 8 it contributes 0. This is FREE in DSP -- input_t feeds the
        # encoder, not the dot multipliers, so the extra width costs wires and a wider
        # shifter, never a multiplier bit.
        INPUT_F = max(0, MANT_F - EXP_MIN + NPELICAN_BFP_GUARD_BITS)
        INPUT_W = INPUT_I + INPUT_F

        # mraw_t: p >> e held EXACTLY (no bits shifted off the bottom), so the ONLY
        # rounding on the encode path is the final cast to mant_t. Right shift by up
        # to EXP_MAX needs EXP_MAX extra fractional bits; the integer part is unchanged.
        MRAW_I = INPUT_I
        MRAW_W = INPUT_W + EXP_MAX
        # mdot_t: the mantissa Minkowski dot, EXACT. mant_t^2 = <2W,4>; summing four
        # of them adds 2 integer bits -> <2W+2,6>. True bound |md| < 4*2*2 = 16 < 2^5,
        # and ap_fixed<.,6> spans +-2^5, so nothing saturates and nothing rounds.
        MDOT_I = 6
        MDOT_W = 2 * MANT_W + 2
        # dotalign_t: md << (e1+e2), still EXACT — the realignment that has to happen
        # BEFORE the dot_t cast, so the whole front end rounds exactly once (at dot_t,
        # AP_RND_CONV), matching PyTorch's single input_quant rounding of the dot.
        ALIGN_F = MDOT_W - MDOT_I
        ALIGN_I = MDOT_I + 2 * EXP_MAX
        ALIGN_W = ALIGN_I + ALIGN_F

        _quant_info['blockfp'].update(
            mant_W=MANT_W, mant_I=MANT_I, exp_bits=EXP_BITS, shift_bits=SHIFT_BITS,
            mraw_W=MRAW_W, mraw_I=MRAW_I, mdot_W=MDOT_W, mdot_I=MDOT_I,
            align_W=ALIGN_W, align_I=ALIGN_I)

        print(f'  (input_t from TRAINED BLOCK-FP pmu grid: mantissa W={MANT_W} I=2 '
              f'(LSB 2^{-MANT_F:+d} of 2^e), exponent [{EXP_MIN},{EXP_MAX}] in '
              f'{EXP_BITS} bits)')
        print(f'   mant_t=ap_fixed<{MANT_W},{MANT_I}>  mdot_t=ap_fixed<{MDOT_W},{MDOT_I}> '
              f'(exact)  dotalign_t=ap_fixed<{ALIGN_W},{ALIGN_I}> (exact)')
        if bfp.get('exp_table') is not None:
            etab = bfp['exp_table']
            print(f'   STATIC per-slot exponent (SPS): exp[slot] = {etab}')
            print(f'   -> clip 2^(e+1) GeV per slot = {[2 ** (v + 1) for v in etab]}')
        print(f'   input_t=ap_fixed<{INPUT_W},{INPUT_I}> is the raw-momentum PORT + '
              f'encoder input only (|p|max={pmax:.1f}); dot multipliers are '
              f'{MANT_W}x{MANT_W}, not {INPUT_W}x{INPUT_W}')
        if args.max_input_bits is not None:
            print('  (--max-input-bits IGNORED under block-FP: the mantissa width is the '
                  'trained knob, and input_t no longer feeds the dot multipliers)')
    elif pmu is not None:
        # --- Phase A* (INPUT_WIDTH_RETRAIN_PLAN.md): the momentum grid was TRAINED
        # (QuantIdentity on Pmu before dot4). input_t IS that learned grid: PyTorch
        # and the firmware quantize momenta identically, so dots are bit-exact by
        # construction — the analytic headroom bound and the Lever-2 cap don't apply.
        if not pmu['signed']:
            sys.exit('ERROR: pmu_quant is unsigned; momenta need a signed type.')
        INPUT_W, INPUT_I, _, INPUT_k = _pt(pmu['scale'], pmu['signed'], pmu['bits'], 'pmu_quant')
        INPUT_F = INPUT_W - INPUT_I
        _pj = act.get('pmu_quant_jet')
        if _pj is not None:
            if not _pj['signed']:
                sys.exit('ERROR: pmu_quant_jet is unsigned; momenta need a signed type.')
            JET_W, JET_I, _, JET_k = _pt(_pj['scale'], _pj['signed'], _pj['bits'], 'pmu_quant_jet')
            print(f'  (jet_t from TRAINED pmu_quant_jet: W={JET_W}, I={JET_I}, '
                  f'momentum LSB = 2^{-(JET_W - JET_I):+d} GeV)')
        print(f'  (input_t from TRAINED pmu_quant: W={INPUT_W}, I={INPUT_I}, '
              f'clip ±2^{INPUT_I - 1} vs |p|max={pmax:.1f} in sample, '
              f'momentum LSB = 2^{-INPUT_F:+d} GeV)')
        if args.max_input_bits is not None:
            print('  (--max-input-bits IGNORED: input_t comes from the trained pmu_quant grid)')
        if args.pmu_extra_int_bits:
            INPUT_I += args.pmu_extra_int_bits
            if not args.pmu_keep_width:
                INPUT_W += args.pmu_extra_int_bits
            INPUT_F = INPUT_W - INPUT_I
            print(f'  (input_t ABLATION --pmu-extra-int-bits {args.pmu_extra_int_bits}: '
                  f'ap_fixed<{INPUT_W},{INPUT_I}>, clip ±2^{INPUT_I - 1}, LSB 2^{-INPUT_F:+d} GeV '
                  f'— NOT the trained grid)')
    else:
        INPUT_I = _int_bits(pmax)
        # Floor at 0: a very coarse dot grid (dot_F <= -(Pbits+3)) genuinely needs no
        # fractional momentum bits, but ap_fixed requires F >= 0 (W >= I).
        INPUT_F = max(0, int(math.ceil(math.log2(pmax))) + dotall_F + 3)   # dotall_F == dot_F w/o split
        INPUT_W = INPUT_I + INPUT_F

        # --- Lever 2: optional cap on input_t width (--max-input-bits). Trades dot4
        # front-end precision for DSP (a narrower mul_NsNs, e.g. <=18 -> 1 DSP). Only
        # INPUT_F is shaved (INPUT_I/range preserved so momenta never saturate); the
        # dots then round to dot_t slightly differently from PyTorch on some events,
        # which the online golden tolerance gate must re-validate. No-op if the cap is
        # >= the bit-exact width.
        if args.max_input_bits is not None and args.max_input_bits < INPUT_W:
            if args.max_input_bits < 2:
                sys.exit(f'ERROR: --max-input-bits={args.max_input_bits} < 2; need at least '
                         f'sign + 1 magnitude bit.')
            # Caps <= INPUT_I are allowed: INPUT_I (range) is ALWAYS preserved so momenta
            # never saturate, and F goes NEGATIVE (ap_fixed permits I > W; e.g. <10,12>
            # stores multiples of 2^2 = 4 GeV). A negative-F cap is hardware-identical to
            # "divide momenta by 2^-F and feed a W-bit integer" — the binary point is free.
            capped_F = args.max_input_bits - INPUT_I
            print(f'  (input_t Lever-2 cap: width {INPUT_W} -> {args.max_input_bits}, '
                  f'F {INPUT_F} -> {capped_F}; momentum LSB = 2^{-capped_F:+d} GeV; '
                  f'bit-exact dots NOT guaranteed — re-check gate)')
            INPUT_F = capped_F
            INPUT_W = args.max_input_bits

    bias_max = float(max(abs(np.asarray(b1)).max(), abs(np.asarray(b1d)).max(), abs(np.asarray(b2)).max()))
    # b1,b1_diag are added in the 2->2 MAC then quantized to the relu grid; b2 in the 2->0
    # MAC then quantized to the out grid. Size F to the FINER of those two learned grids so
    # the bias-rounding error (2^-(BIAS_F+1)) stays <= 1/4 LSB of whichever grid it feeds;
    # +1 = guard bit. Derived from the learned scales so it tracks the QAT bit-width flags
    # (was a hardcoded 24, sized for the old 24-bit grids).
    BIAS_F = max(relu_F, out_F) + 1          # legacy single-grid rule (kept for the log only)
    BIAS_I = _int_bits(bias_max)
    BIAS_W = BIAS_I + BIAS_F
    # Per-stage bias types (2026-10-06): a bias is added to a MAC sum whose grid is the exact
    # product grid (w_F + t_F). The old max(relu_F,out_F)+1 rule left up to 1/4 LSB of bias
    # error (6-bit 5-class ckpt: b2=-0.0053 -> 0, 146/200 golden mismatches, all in Rp).
    # Snapping the bias onto the product grid itself is still not enough: the products are
    # exact on that grid, so a grid-rounded bias can land the pre-quantizer sum EXACTLY on a
    # half-LSB tie of the next quantizer (ev 34: Rp=0.21875=3.5 LSB -> RNE 0.25, while torch's
    # float 0.218444 -> 0.1875). G = --bias-guard-bits extra fractional bits keep the bias off
    # that grid (error <= 2^-(w_F+t_F+G+1)), so ties are as unlikely as in float32.
    BIAS_G = args.bias_guard_bits
    BIAS1_F = (w1_W - w1_I) + (t2_W - t2_I) + BIAS_G
    BIAS2_F = (w2_W - w2_I) + (t0_W - t0_I) + BIAS_G
    BIAS1_I = _int_bits(float(max(abs(np.asarray(b1)).max(), abs(np.asarray(b1d)).max())))
    BIAS2_I = _int_bits(float(abs(np.asarray(b2)).max()))
    BIAS1_W, BIAS2_W = BIAS1_I + BIAS1_F, BIAS2_I + BIAS2_F

    bn_max = float(np.abs(np.asarray(batch1)).max())
    bn_max = max(bn_max, float(np.abs(np.asarray(batch2)).max()))
    # The BN scale multiplies (dots - mean); size its fractional part so the scale-rounding
    # error stays under half the t2 LSB even for dots spanning the full dot_t range
    # (data-independent / robust to the learned mean). +2 is margin.
    BN_I = _int_bits(bn_max)
    # BN guard bits (2026-10-07, default 0 = unchanged resource behaviour): the BN1 scale literal is
    # DSP-bound and deliberately short (see --bn-frac-bits); its relative error (6.6e-5 at 13 bits)
    # times a large dot sum can flip a post-agg rounding tie (measured: a jdotp 1.7e-5 LSB from the
    # tie on a 6-bit 5-class jet checkpoint -> 199/200). G extra bits buy exactness at DSP cost.
    BN_F = t2_F + dotall_mag + 2 + int(args.bn_guard_bits)      # dotall_mag == dot_mag without --jet-quant-split
    # --- BN1 DSP threshold cap (--bn-frac-bits) ---
    # The BN1 scale gamma/sigma is a SCALAR constant multiplying every dot, so its snapped
    # literal width decides the binding of NPARTICLES2*(NPARTICLES2+1)/2 multipliers at once.
    # Measured on xcu250 @5ns (2026-08-25): a 10-bit literal (653, 5 CSD terms) binds a
    # DSP48 per multiply AND violates timing (slack -0.00); an 8-bit literal (163, 4 terms)
    # lands in fabric at ~49 LUT each and clears timing. 9 bits is untested. At 16
    # particles that swing is 171 DSP for ~1.5k LUT -- the cheapest DSP lever measured.
    # Latency does NOT recover (15 cycles either way); only the timing violation was BN1's.
    # Bind Op module names show the literal width PLUS a zero-extension bit for the signed
    # multiplier (653 -> mul_6s_11ns, 163 -> mul_6s_9ns) -- do not read the module name as
    # the literal width. Lowering BN_F shifts the literal right; check the snap-error print
    # (budget: half the t2 LSB through the multiply; the derived BN_F carries +2 margin).
    if args.bn_frac_bits is not None:
        if args.bn_frac_bits > BN_F:
            raise SystemExit(f'--bn-frac-bits {args.bn_frac_bits} exceeds the derived '
                             f'BN_F={BN_F}; the cap may only reduce it.')
        BN_F = int(args.bn_frac_bits)
    BN_W = BN_I + BN_F
    # Report the snapped BN1 scale literal so the DSP binding is visible without synthesis.
    _bn1_scale = float(np.asarray(batch1).reshape(-1)[1])
    _bn1_lit = round(_bn1_scale * (2 ** BN_F))
    _bn1_bits = int(_bn1_lit).bit_length()
    _bn1_err = abs(_bn1_lit / (2 ** BN_F) - _bn1_scale)
    print(f'  BN1 scale gamma/sigma = {_bn1_scale:.12g} -> literal {_bn1_lit} at F={BN_F} '
          f'({_bn1_bits} bits, snap err {_bn1_err:.3g})')
    _verdict = ('fabric (8-bit measured in fabric)' if _bn1_bits <= 8 else
                'UNTESTED (9 bits: fabric measured at 8, DSP48 at 10)' if _bn1_bits == 9 else
                'DSP48 + timing risk (10-bit measured on DSP) -- see --bn-frac-bits')
    print(f'   BN1 multiplier binding: {_verdict}')

    # --- accumulator headroom from NPARTICLES2 (NPARTICLES + 2 spurions).
    # NPARTICLES2 is a firmware constant the loader already mirrors (NPELICAN.h:
    # NPARTICLES2 = NPARTICLES + 2 = 22); accumulators are NOT covered by any
    # learned scale, so they get explicit integer headroom over the summand type.
    # (legacy rule: sized for NPARTICLES=20 regardless of the checkpoint's nobj; the jet
    # spurion adds one row/col -> 20 + NSPURIONS.)
    # Accumulator headroom follows the CHECKPOINT's particle count (fallback 20): sums run over
    # NPARTICLES2^2 (full) / NPARTICLES2 (row) terms, and the acc types have no AP_SAT, so an
    # under-sized accumulator (e.g. a 20-slot rule applied to an N=32 export) would wrap silently.
    NPARTICLES = int(_arg('nobj')) if _arg('nobj', None) is not None else 20
    NPARTICLES2 = NPARTICLES + NSPURIONS
    H2 = math.ceil(math.log2(NPARTICLES2 ** 2))   # full-sum headroom = 9
    H1 = math.ceil(math.log2(NPARTICLES2))        # row-sum headroom  = 5

    # --- Aggregation summands (batch1 for 2->2, Tr for 2->0) are the UNQUANTIZED BatchNorm
    # outputs that PyTorch sums in float; only the NORMALIZED result hits a learned quantizer.
    # So each summand is typed by its OWN range (not the downstream quantizer's I), and gets a
    # generous fractional width AGG_F: storing it on the coarse post-agg grid (t2_F=18) would
    # leave a ~2^-19 per-term error that, summed and renormalized, tips the post-agg rounding
    # boundary and breaks bit-exactness. AGG_F exceeds the finest post-agg grid so neither
    # path's rounding tips (normalized rounding error ~ 2^-(AGG_F+2.3) << half the t0 LSB).
    # Aggregation guard bits (2026-10-06): batch1 is SUMMED over up to NPARTICLES2 entries
    # before the single x1/Nbar rescale and t2 rounding, so its storage grid must be set by
    # the SUM's error budget, not the per-element grid: N2 terms x 2^-(AGG_F+1) each, /Nbar,
    # must stay far below half a t2 LSB or the rescaled sum tips a rounding tie (seen on a
    # 6-bit checkpoint: jdotp row sum at exactly -0.25 = -1/2 LSB -> firmware 0, PyTorch -0.5).
    # G extra bits make that error 2^-G smaller; same idea as --bias-guard-bits.
    AGG_F = max(t2_W - t2_I, t0_W - t0_I) + 1 + int(args.agg_guard_bits)   # = max(t2_F, t0_F) + 1 + G

    # batch1 = BatchNorm1(dots): range bound from the dot_t span and the BN1 constants.
    bn1 = np.asarray(batch1).reshape(-1)                 # [mean, scale, beta]
    dot_max = 2.0 ** dotall_mag   # dots span dotall_t (jet dots / m_jet^2 under the split)
    bn1_bound = (dot_max + abs(bn1[0])) * abs(bn1[1]) + abs(bn1[2])
    bn1_I = int(math.ceil(math.log2(bn1_bound))) + 1     # +1 sign bit
    bn1_W = bn1_I + AGG_F

    # Tr = BatchNorm2(relu): range blown up ~130x by the BN2 scale, so it needs a wide I or it
    # saturates (the post_agg-2to0 quantizer's I=1 reflects only the normalized |R|<1, not Tr).
    relu_max = 2.0 ** (relu_I - 1)
    bn2 = np.asarray(batch2).reshape(-1, 3)              # [channel][mean, scale, beta]
    tr_bound = float(max((relu_max + abs(m)) * abs(s) + abs(b) for m, s, b in bn2))
    tr_I = int(math.ceil(math.log2(tr_bound))) + 1       # +1 sign bit
    tr_W = tr_I + AGG_F

    # Accumulators size from their actual summand type, NOT t2_t / t0_t.
    # 2->2 path sums bn1out_t (batch1). 2->0 path: BN2 is collapsed PAST the aggregation
    # (firmware #2), so the 2->0 accumulators now sum the ReLU output relu_t (Tp_q) raw —
    # NOT tr_t. The per-channel BN2 affine (s·A + β'·count) is applied once to the aggregate.
    acc2_W, acc2_I             = bn1_W + H2, bn1_I + H2
    accrow_W, accrow_I         = bn1_W + H1, bn1_I + H1
    accrelu_W, accrelu_I       = relu_W + H2, relu_I + H2   # R full sum of relu_t Tp_q
    accrelurow_W, accrelurow_I = relu_W + H1, relu_I + H1   # R trace, sum of relu_t Tp_q
    # The value the 2->0 norm_t multiplies is the BN2-applied full sum (s·A + β'·count),
    # i.e. the sum-of-Tr magnitude = tr_I + H2 integer bits — same as before the collapse,
    # only its computation moved. Used solely to size NORM_F's rounding bound below.
    agg0_I = tr_I + H2

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
    # mac0_t also receives b2 (F = w2_F+t0_F+G): widen F to BIAS2_F so that add is exact.
    mac0_W = mac0_I + BIAS2_F
    # mac2_t is NOT widened (NPARTICLES2^2*NHIDDEN accumulators x 8 terms); the bias is added
    # once per element in mac2b_t (same I, F = BIAS1_F) just before the relu quantizer.
    mac2b_I = mac2_I
    mac2b_W = mac2b_I + BIAS1_F

    # --- Nonlinear head (--head-hidden K): agg_2to0.act_layer quantizer -> relu0_t, head
    # weight quantizer -> wh_gen_t; float head bias with guard bits on the head product grid
    # (biash_t_gen, same G as bias2_t_gen); head MAC mach_t with exact products + exact
    # bias add (like mac0_t): I = wh_I + relu0_I + ceil(log2(K+1)), F = BIASH_F.
    head = None
    if HEAD > 0:
        _ra = act['agg_2to0.act_layer']
        r0_W, r0_I, r0_s, r0_k = _pt(_ra['scale'], _ra['signed'], _ra['bits'], 'agg_2to0.act_layer (head ReLU)')
        _wh = wgt['head']
        wh_W, wh_I, wh_s, wh_k = _pt(_wh['scale'], _wh['signed'], _wh['bits'], 'head weights')
        b_head = _quant_info['head']['b']
        BIASH_F = (wh_W - wh_I) + (r0_W - r0_I) + BIAS_G
        BIASH_I = _int_bits(float(np.abs(np.asarray(b_head)).max()))
        BIASH_W = BIASH_I + BIASH_F
        mach_terms = HEAD + 1             # K products wh*relu0 + bias
        mach_I = wh_I + r0_I + math.ceil(math.log2(mach_terms))
        mach_W = mach_I + BIASH_F
        head = dict(r0=(r0_W, r0_I, r0_s, r0_k, _ra['scale']), wh=(wh_W, wh_I, wh_s, wh_k, _wh['scale']),
                    biash=(BIASH_W, BIASH_I, BIASH_F), mach=(mach_W, mach_I, mach_terms))

    def _fixed(W, I, signed, rnd='AP_RND_CONV', sat='AP_SAT'):
        base = 'ap_fixed' if signed else 'ap_ufixed'
        return f'{base}<{W}, {I}, {rnd}, {sat}>'

    L = []
    L.append('#ifndef NPELICAN_TYPES_GENERATED_H_')
    L.append('#define NPELICAN_TYPES_GENERATED_H_')
    L.append('')
    L.append('// Model dimensions read from the checkpoint. nPELICAN.h guards its hand')
    L.append('// defaults with #ifndef, so these take precedence (NPARTICLES2 = NPARTICLES + NSPURIONS).')
    _nobj = _arg('nobj', None)
    if _nobj is not None:
        L.append(f'#define NPARTICLES {int(_nobj)}')
    else:
        L.append('// checkpoint args.nobj was None: NPARTICLES keeps the nPELICAN.h hand default')
    L.append(f'#define NHIDDEN {NHIDDEN}')
    L.append(f'#define NOUT {NOUT}')
    L.append(f'#define NSPURIONS {NSPURIONS}  // 2 beams' + (' + full-jet spurion (args.add_jet)' if NSPURIONS == 3 else ''))
    if NSPURIONS == 3:
        L.append('#define NPELICAN_JET_SPURION 1  // top gains jet_t jet_input[4] -> slot 2')
    if HEAD > 0:
        L.append(f'#define NPELICAN_HEAD {HEAD}  // args.head_hidden: 2->0 -> K ReLU -> K->NOUT head')
        L.append(f'#define N2TO0_OUT {HEAD}')
    L.append('')
    L.append('// GENERATED by model_loader.py --quant. Do not hand-edit.')
    L.append('// Per-quantizer fixed-point types derived from the checkpoint\'s learned')
    L.append('// Brevitas scales (k = -log2(scale); I = bits - k). Accumulator / MAC types')
    L.append('// derived by formula from those + term counts. Phase 1: INERT include — not')
    L.append('// yet wired into the datapath (Phase 2 swaps usage and retires the old types).')
    L.append('')
    L.append('#include "ap_fixed.h"')
    L.append('#include "ap_int.h"')
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
    # result_t (the firmware's final-logit output type) MUST equal out_t (the
    # output_quant grid). It was hand-fixed to ap_fixed<24,1> in nPELICAN.h, which
    # silently CLAMPS the logit to [-1,1) for any checkpoint whose output_quant range
    # is wider (e.g. out_t=ap_fixed<W,3> -> [-4,4)); that compresses the score ranking
    # and wrecks AUC. Generate it = out_t (guarded, like input_t) so the firmware
    # preserves the model's full logit range for every bit width.
    L.append('#define NPELICAN_RESULT_T_GENERATED 1')
    L.append(f'typedef {_fixed(out_W, out_I, out_s)} result_t;  // == out_t (output_quant grid)')
    L.append(_pt_line('w1_gen_t', w1_W,   w1_I,   w1_s,   wgt['net2to2.eq_layers.0']['scale'], w1_k, '2->2 weights'))
    L.append(_pt_line('w2_gen_t', w2_W,   w2_I,   w2_s,   wgt['agg_2to0']['scale'], w2_k, '2->0 weights'))
    if head is not None:
        _r0, _wh = head['r0'], head['wh']
        L.append(_pt_line('relu0_t', _r0[0], _r0[1], _r0[2], _r0[4], _r0[3], 'agg_2to0.act_layer (head QuantReLU)'))
        L.append(_pt_line('wh_gen_t', _wh[0], _wh[1], _wh[2], _wh[4], _wh[3], 'head weights'))
    L.append('')
    L.append('// ---- Raw-momentum / IO interface type (input_t): operand of the dot4')
    if bfp is not None:
        L.append('//      encoder, NOT of the dot multipliers. Lever 7 block-FP: the momentum')
        L.append('//      grid is PER-PARTICLE relative (p_k = m_k * 2^e), so no single uniform')
        L.append('//      type describes it. input_t carries the raw momenta to the on-chip')
        L.append('//      encoder (firmware/np_blockfp.h); the multipliers are mant_t x mant_t.')
        L.append('//      F is sized so the port rounding sits >=2 bits under the FINEST')
        L.append('//      mantissa LSB (at e=EXP_MIN), i.e. the encode lands where PyTorch\'s')
        L.append('//      single rounding of the float momenta lands.')
    elif pmu is not None:
        L.append('//      multipliers. TRAINED grid (Phase A*): pmu_quant = QuantIdentity on Pmu')
        L.append('//      before dot4, learned po2 scale. PyTorch and firmware grid the momenta')
        L.append('//      identically, so dots are bit-exact by construction (no analytic bound,')
        L.append('//      no Lever-2 cap). Leading |p| above the clip point saturates BY DESIGN;')
        L.append('//      the model was trained through that clipping.')
    else:
        L.append('//      multipliers (36x36 today) that dominate DSP. NOT a learned quantizer;')
        L.append('//      input_quant grids the DOTS (dot_t). I = physics |p| range (flag-')
        L.append('//      independent); F = ceil(log2|p|max) + dot_F + 3 so the dot4 product error')
        L.append('//      stays < 1/2 dot_t LSB (bit-exact dots). F tracks the dot grid, so lower')
        L.append('//      QAT bits -> smaller F -> narrower input_t -> cheaper dot multipliers.')
    L.append('//      Guard macro lets nPELICAN.h keep a hand fallback for the float path.')
    L.append('#define NPELICAN_INPUT_T_GENERATED 1')
    if bfp is not None:
        L.append(f'typedef ap_fixed<{INPUT_W}, {INPUT_I}, AP_RND_CONV, AP_SAT> input_t;'
                 f'  // raw momenta into the block-FP encoder; |p|max={pmax:.1f} '
                 f'(I={INPUT_I}), F={INPUT_F}')
    elif pmu is not None:
        L.append(f'typedef ap_fixed<{INPUT_W}, {INPUT_I}, AP_RND_CONV, AP_SAT> input_t;'
                 f'  // raw momenta; TRAINED pmu_quant grid (I={INPUT_I}, F={INPUT_F}); '
                 f'|p|max={pmax:.1f}'
                 + (f'; ABLATION --pmu-extra-int-bits {args.pmu_extra_int_bits}'
                    f'{" --pmu-keep-width" if args.pmu_keep_width else ""} (NOT deployable)'
                    if args.pmu_extra_int_bits else ''))
    else:
        L.append(f'typedef ap_fixed<{INPUT_W}, {INPUT_I}, AP_RND_CONV, AP_SAT> input_t;'
                 f'  // raw momenta; |p|max={pmax:.1f} (I={INPUT_I}), F={INPUT_F} (dot_F={dot_F})')
    L.append('')
    L.append('// ---- Jet-spurion quantizer split (--jet-quant-split). Without it these alias the')
    L.append('//      single-grid types, so the firmware can always name them (byte-identical).')
    L.append('#define NPELICAN_JET_TYPES_GENERATED 1')
    if JET_QUANT_SPLIT:
        L.append('#define NPELICAN_JET_QUANT_SPLIT 1  // args.jet_quant_split: d[2,j]/d[i,2] -> dotj_t, d[2,2] -> dotm_t')
        L.append(_pt_line('dotj_t', dotj_W, dotj_I, dotj_s, act['input_quant_jet']['scale'], dotj_k, 'input_quant_jet'))
        L.append(_pt_line('dotm_t', dotm_W, dotm_I, dotm_s, act['input_quant_mjet']['scale'], dotm_k, 'input_quant_mjet'))
        L.append(f'typedef {_fixed(dotall_W, dotall_I, dot_s)} dotall_t;'
                 f'  // dots array: I = max I, F = max F of dot_t/dotj_t/dotm_t (exact container)')
        if _pj is not None:
            L.append(f'typedef ap_fixed<{JET_W}, {JET_I}, AP_RND_CONV, AP_SAT> jet_t;'
                     f'  // jet_input port: TRAINED pmu_quant_jet grid, scale=2^-{JET_k} '
                     f'({_pj["scale"]:.9e}), bits={JET_W}')
        else:
            L.append('typedef input_t jet_t;  // no trained momentum grid: jet shares input_t')
    else:
        L.append('typedef dot_t dotj_t;')
        L.append('typedef dot_t dotm_t;')
        L.append('typedef dot_t dotall_t;')
        L.append('typedef input_t jet_t;')
    L.append('')

    if bfp is not None:
        L.append('// ---- Lever 7: per-particle BLOCK FLOATING POINT momentum grid ----------')
        L.append('// p_k = m_k * 2^e, ONE exponent per particle, shared by its 4 components:')
        if bfp.get('exp_table') is None:
            L.append('//   e   = clamp(floor(log2|E|), EXP_MIN, EXP_MAX)   (one LZC on the energy)')
        else:
            L.append('//   e   = NPELICAN_BFP_EXP_TABLE[slot]   (SPS: trained constant per slot, below)')
        L.append(f'//   m_k = p_k * 2^-e on a signed {MANT_W}-bit I=2 grid (LSB 2^{-MANT_F:+d} of 2^e)')
        if bfp.get('exp_table') is None:
            L.append('// I=2 cannot overflow: E >= |p_k| and 2^e <= E < 2^(e+1) => |m_k| < 2.')
        else:
            L.append('// SPS: |p_k| >= 2^(e+1) in a slot SATURATES m_k (AP_SAT), exactly as PyTorch clamps.')
        L.append('// Masking invariant holds: a zeroed particle gives e = EXP_MIN, m = 0, dot 0.')
        L.append('// The dot is d = 2^(e1+e2) * (m1 . g . m2): mantissa dot EXACT in mdot_t,')
        L.append('// realigned EXACTLY in dotalign_t, then rounded ONCE into dot_t (AP_RND_CONV)')
        L.append('// -- the same single rounding PyTorch applies with input_quant.')
        L.append('#define NPELICAN_BLOCK_FP 1')
        L.append(f'#define NPELICAN_BFP_MANT_W {MANT_W}')
        L.append(f'#define NPELICAN_BFP_EXP_MIN {EXP_MIN}')
        L.append(f'#define NPELICAN_BFP_EXP_MAX {EXP_MAX}')
        if bfp.get('exp_table') is not None:
            etab = bfp['exp_table']
            L.append('// SPS: STATIC per-slot exponent. The exponent above is NOT computed at run')
            L.append('// time: e[slot] is a TRAINED constant (PELICAN-nano BlockFPQuant static mode,')
            L.append('// exponent_table()), so the encoder/realign shifts are wiring. Slot order =')
            L.append('// firmware P1Prep: 0,1 = beam spurions, 2..21 = constituents in input order.')
            L.append('// Per-slot momentum clip 2^(e+1) GeV (mantissa range [-2,2) x 2^e):')
            L.append('//   ' + ', '.join(f'[{i}] {2 ** (v + 1)}' for i, v in enumerate(etab)))
            L.append('// Sized with a literal because NPARTICLES2 is defined in nPELICAN.h AFTER')
            L.append('// this header is included; np_blockfp.h static_asserts it equals NPARTICLES2.')
            L.append('#define NPELICAN_BFP_STATIC 1')
            L.append(f'static const int NPELICAN_BFP_EXP_TABLE[{len(etab)}] = '
                     '{ ' + ', '.join(str(v) for v in etab) + ' };')
        L.append(f'typedef ap_fixed<{MANT_W}, {MANT_I}, AP_RND_CONV, AP_SAT> mant_t;'
                 f'  // mantissa; the dot multiplier operand ({MANT_W}x{MANT_W})')
        L.append(f'typedef ap_uint<{EXP_BITS}> bexp_t;'
                 f'  // per-particle exponent, [{EXP_MIN},{EXP_MAX}]')
        L.append(f'typedef ap_uint<{SHIFT_BITS}> bshift_t;'
                 f'  // e1+e2 realign amount, [{2*EXP_MIN},{2*EXP_MAX}]')
        L.append(f'typedef ap_fixed<{MRAW_W}, {MRAW_I}> mraw_t;'
                 f'  // p >> e held EXACTLY (F = input F + EXP_MAX), so the encode rounds'
                 f' only once, at the mant_t cast')
        L.append(f'typedef ap_fixed<{MDOT_W}, {MDOT_I}> mdot_t;'
                 f'  // EXACT mantissa Minkowski dot: <2W,4> products, +2 int bits for the'
                 f' 4-term sum; |md| < 16')
        L.append(f'typedef ap_fixed<{ALIGN_W}, {ALIGN_I}> dotalign_t;'
                 f'  // EXACT mdot << (e1+e2); I = {MDOT_I} + 2*EXP_MAX so the realign never'
                 f' saturates')
        L.append('')
    L.append('// ---- Float-trained biases / BatchNorm constants / normalization constants ----')
    L.append('// These are NOT PyTorch quantization points (PyTorch keeps them in float), so')
    L.append('// per CLAUDE.md/plan they are WIDENED, not snapped: their fixed-point rounding')
    L.append('// error must stay below half the LSB of the next real quantizer they feed.')
    # bias_t_gen: biases feed the dense MAC then the relu/out quantizers (2^-22 / 2^-23);
    # F=24 keeps |err|<=2^-25, and the integer width is derived from |bias|max (above).
    L.append(f'typedef ap_fixed<{BIAS1_W}, {BIAS1_I}, AP_RND_CONV, AP_SAT> bias1_t_gen;'
             f'  // b1,b1_diag (float): F = mac2 grid w1_F+t2_F + G = {BIAS1_F} (G={BIAS_G} guard bits); |b|max I={BIAS1_I}')
    L.append(f'typedef ap_fixed<{BIAS2_W}, {BIAS2_I}, AP_RND_CONV, AP_SAT> bias2_t_gen;'
             f'  // b2 (float): F = mac0 grid w2_F+t0_F + G = {BIAS2_F} (G={BIAS_G} guard bits); |b|max I={BIAS2_I}')
    L.append('typedef bias1_t_gen bias_t_gen;  // legacy alias (single bias type, pre-2026-10-06)')
    if head is not None:
        _bw, _bi, _bf = head['biash']
        L.append(f'typedef ap_fixed<{_bw}, {_bi}, AP_RND_CONV, AP_SAT> biash_t_gen;'
                 f'  // b_head (float): F = head grid wh_F+relu0_F + G = {_bf} (G={BIAS_G} guard bits); |b|max I={_bi}')
    # bn_t_gen: BN mean/scale/beta. The scale gamma/sigma multiplies (dots-mean), so F is sized
    # to keep scale-rounding under half the t2 LSB across the full dot_t range; I is derived
    # from |c|max (the running_mean grows with jet energy and is the usual driver).
    L.append(f'typedef ap_fixed<{BN_W}, {BN_I}, AP_RND_CONV, AP_SAT> bn_t_gen;'
             f'  // BN mean/scale/beta (float); |c|max={bn_max:.6g} (I={BN_I}), F={BN_F}')
    # norm_t: invnave=1/N̄, invnave2=1/N̄^2 (not po2). A 12-frac internal_t mis-rounds invnave2
    # by ~40%. They multiply the raw aggregation sums (mag ~ 2^acc_I) and the normalized result
    # lands on the post-agg grids: the 2->2 accumulator (acc2_I) renormalizes onto the t2 grid,
    # the 2->0 full sum (agg0_I = tr_I+H2) onto the t0 grid. For each path the norm-rounding error
    # 2^-(NORM_F+1) scaled by the accumulator must stay <= 1/4 LSB of that path's grid:
    #   2^acc_I * 2^-(NORM_F+1) <= 2^-(postF+2)  =>  NORM_F >= acc_I + postF + 1.
    # One shared norm_t covers both, so take the max over the two paths (full-sum accumulators
    # dominate their row-sum counterparts, H2>H1). Derived from the learned scales + accumulator
    # widths so it tracks the QAT bit-width flags (was a hardcoded <40,1> sized for the old
    # grids, which was in fact 2 bits short at t0_F=23). NORM_I=1: 1/N̄,1/N̄^2 are in [0,1).
    NORM_I = 1
    # Normalization guard bits (2026-10-07): 1/Nbar and 1/Nbar^2 are NOT powers of two, so the
    # norm_t literal carries a relative error ~2^-(NORM_F) / value. That error moves every
    # rescaled sum by up to |R|*err and flips any post-agg rounding tie closer than that
    # (measured on a 6-bit 5-class ckpt: F=26 -> invnave2 rel err 1.4e-5 moved -9.500062 LSB to
    # -9.499933 -> 1-LSB logit mismatch). G extra bits cost two constant multipliers per channel.
    NORM_F = max(acc2_I + t2_F, agg0_I + t0_F) + 1 + int(args.norm_guard_bits)
    NORM_W = NORM_I + NORM_F
    L.append(f'typedef ap_fixed<{NORM_W}, {NORM_I}, AP_RND_CONV, AP_SAT> norm_t;'
             f'  // 1/N̄, 1/N̄^2 normalize-late multipliers (F={NORM_W-NORM_I})')
    L.append('')
    L.append('// ---- Aggregation summands (unquantized BatchNorm outputs) + their accumulators.')
    L.append(f'//      AGG_F = {AGG_F} fractional bits (> the finest post-agg grid t0_F={t0_W-t0_I}) so the')
    L.append('//      raw-sum-then-renormalize lands bit-exactly on the post-agg quantizer grid;')
    L.append('//      storing these on the coarse post-agg grid (e.g. t2_F) would tip the rounding.')
    L.append('//      Each is range-typed by its OWN BN output bound; SAT guards it. Accumulators')
    L.append(f'//      add ceil(log2(#terms)) integer headroom (H2={H2} full sum, H1={H1} row/trace).')
    L.append(f'typedef ap_fixed<{bn1_W}, {bn1_I}, AP_RND_CONV, AP_SAT> bn1out_t;'
             f'  // batch1 = BN1(dots); |batch1|<={bn1_bound:.1f} (I={bn1_I}), F={AGG_F}')
    L.append(f'typedef ap_fixed<{acc2_W}, {acc2_I}> acc2_t;     // jmass raw sum of bn1out_t (I(bn1out)+H2)')
    L.append(f'typedef ap_fixed<{accrow_W}, {accrow_I}> accrow_t;   // jdotp row sums of bn1out_t (I(bn1out)+H1)')
    L.append(f'typedef ap_fixed<{tr_W}, {tr_I}, AP_RND_CONV, AP_SAT> tr_t;'
             f'  // Tr = BN2(relu); |Tr|<={tr_bound:.1f} (I={tr_I}); dump-only after #2 (BN2 folded past the 2->0 aggregation)')
    relu_sg = 'ap_fixed' if relu_s else 'ap_ufixed'
    L.append(f'typedef {relu_sg}<{accrelu_W}, {accrelu_I}> accrelu_t;     // R full sum of relu_t Tp_q (I(relu)+H2)')
    L.append(f'typedef {relu_sg}<{accrelurow_W}, {accrelurow_I}> accrelurow_t;  // R trace, sum of relu_t Tp_q (I(relu)+H1)')
    L.append('')
    L.append('// ---- MAC temporaries: I = I(weight)+I(operand)+ceil(log2(#terms)), W = I+B ----')
    L.append(f'typedef ap_fixed<{mac2_W}, {mac2_I}> mac2_t;     // 2->2 dense: 6 w1*t2 products + b1 + b1_diag = {mac2_terms} terms')
    L.append('// bias add for the 2->2 dense output: products accumulate in mac2_t (exact product grid);')
    L.append('// the float-typed bias is added ONCE here at full bias precision, then the relu quantizer rounds')
    L.append('#define NPELICAN_MAC2B_T_GENERATED 1')
    L.append(f'typedef ap_fixed<{mac2b_W}, {mac2b_I}> mac2b_t;    // I = I(mac2_t), F = BIAS1_F = {BIAS1_F}')
    L.append(f'typedef ap_fixed<{mac0_W}, {mac0_I}> mac0_t;     // 2->0 dense: 2*NHIDDEN w2*t0 products + b2 = {mac0_terms} terms; F = BIAS2_F = {BIAS2_F} (exact bias add)')
    if head is not None:
        _mw, _mi, _mt = head['mach']
        L.append(f'typedef ap_fixed<{_mw}, {_mi}> mach_t;     // head: K={HEAD} wh*relu0 products + b_head = {_mt} terms; F = BIASH_F = {head["biash"][2]} (exact bias add)')
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
    bn_type, bias_type, norm_type = 'bn_t_gen', 'bias1_t_gen', 'norm_t'
    bias2_type = 'bias2_t_gen'
else:
    w1_type = 'w1_t' if args.split_types else 'weight_t'
    w2_type = 'w2_t' if args.split_types else 'weight_t'
    bn_type, bias_type, norm_type = 'weight_t', 'bias_t', 'internal_t'
    bias2_type = 'bias_t'

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
    _nobj_sa = _arg('nobj', None)
    _sa_cond = f'NHIDDEN == {NHIDDEN} && NOUT == {NOUT} && NSPURIONS == {NSPURIONS}'
    if HEAD > 0:
        _sa_cond += f' && NPELICAN_HEAD == {HEAD} && N2TO0_OUT == {HEAD}'
    if _nobj_sa is not None:
        _sa_cond += f' && NPARTICLES == {int(_nobj_sa)}'
    f.write(f'static_assert({_sa_cond}, "weights.h was exported for NHIDDEN={NHIDDEN} '
            f'NOUT={NOUT} NPARTICLES={_nobj_sa if _nobj_sa is not None else "?"}; '
            f'firmware/nPELICAN.h or types_generated.h disagree -- re-export or fix the defines");\n')
    if NSPURIONS == 3:
        # jet-quant-split presence must match the header (static_assert cannot test a macro)
        if JET_QUANT_SPLIT:
            f.write('#ifndef NPELICAN_JET_QUANT_SPLIT\n#error "weights.h was exported for a '
                    '--jet-quant-split checkpoint but NPELICAN_JET_QUANT_SPLIT is not defined"\n#endif\n')
        else:
            f.write('#ifdef NPELICAN_JET_QUANT_SPLIT\n#error "weights.h was exported WITHOUT '
                    '--jet-quant-split but NPELICAN_JET_QUANT_SPLIT is defined"\n#endif\n')
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
    f.write(f'{w2_type} w2_2to0[N2TO0_OUT*NHIDDEN*2] = ' + _c(w2_2to0) + ';\n')
    f.write(f'{bias2_type} b2_2to0[N2TO0_OUT] = ' + _c(b2_2to0) + ';\n')
    if HEAD > 0:
        f.write('\n//nonlinear head (K -> NOUT), row-major (NOUT, K): w_head[o*NPELICAN_HEAD + k]\n')
        f.write('wh_gen_t w_head[NOUT*NPELICAN_HEAD] = ' + _c(_quant_info['head']['w']) + ';\n')
        f.write('biash_t_gen b_head[NOUT] = ' + _c(_quant_info['head']['b']) + ';\n')

    # D3: measured QAT scales appended as comments for the record (quant path only).
    if _scale_comment_lines:
        f.write('\n'.join(_scale_comment_lines) + '\n')

print(f'\nWrote {args.out}  (NHIDDEN={NHIDDEN}, NOUT={NOUT}, NSPURIONS={NSPURIONS}'
      f'{" [jet spurion]" if NSPURIONS == 3 else ""}'
      f'{" [jet quant split]" if JET_QUANT_SPLIT else ""}, head_hidden={HEAD}, '
      f'NPARTICLES={_arg("nobj", None) if _arg("nobj", None) is not None else "default"}, '
      f'agg guard bits = {args.agg_guard_bits}, norm guard bits = {args.norm_guard_bits}, bn guard bits = {args.bn_guard_bits}, bias guard bits = {args.bias_guard_bits})')
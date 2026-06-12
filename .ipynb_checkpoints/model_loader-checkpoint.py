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
args = parser.parse_args()

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
def _extract_quant_weights():
    if fmt == 'old':
        sys.exit('ERROR: --quant requires a post-refactor checkpoint (mixing.weight keys). '
                 'Convert it first with scripts/convert_checkpoint.py.')

    wbw = args.weight_bit_width or _arg('weight_bit_width', 24)
    abw = args.act_bit_width    or _arg('act_bit_width', 24)
    ibw = args.input_bit_width  or _arg('input_bit_width', 24)
    po2 = (not args.no_po2) if margs is None else _arg('po2_scales', not args.no_po2)

    qcfg = QuantConfig(enabled=True, weight_bit_width=wbw, act_bit_width=abw,
                       input_bit_width=ibw, po2_scales=po2)
    model = PELICANNano(NHIDDEN, quant_config=qcfg,
                        batchnorm=_arg('batchnorm', 'b'),
                        activation=_arg('activation', 'relu'))
    model.load_state_dict(sd)
    model.eval()

    print(f'\nQuant config: weight/act/input bits = {wbw}/{abw}/{ibw}, po2={po2}')
    print('QuantLinear weight scales:')
    qw1 = model.net2to2.eq_layers[0].mixing.quant_weight()
    qw2 = model.agg_2to0.mixing.quant_weight()
    for name, qw in (('net2to2.eq_layers.0', qw1), ('agg_2to0', qw2)):
        s = float(qw.scale)
        print(f'  {name:<22} scale = {s:.6e}  ~ 2^{math.log2(s):.2f}'
              f'  => {-math.log2(s):.0f} fractional bits')

    w1  = np.ravel(qw1.value.detach().numpy())
    w2  = np.ravel(qw2.value.detach().numpy())
    b1  = sd['net2to2.eq_layers.0.bias'].numpy()
    b1d = sd['net2to2.eq_layers.0.diag_bias'].numpy()
    b2  = sd['agg_2to0.mixing.bias'].numpy()
    return w1, b1, b1d, w2, b2

# ---------------------------------------------------------------------------
# Dispatch + write weights.h (format unchanged apart from optional types)
# ---------------------------------------------------------------------------
if args.quant:
    print('Extracting snapped dequantized weights (QAT path, via Brevitas)...')
    w1_2to2, b1_2to2, b1d_2to2, w2_2to0, b2_2to0 = _extract_quant_weights()
else:
    w1_2to2, b1_2to2, b1d_2to2, w2_2to0, b2_2to0 = _extract_float_weights()

w1_type = 'w1_t' if args.split_types else 'weight_t'
w2_type = 'w2_t' if args.split_types else 'weight_t'

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w') as f:
    f.write('#include "../nPELICAN.h"\n')
    f.write('//model: ' + str(_arg('prefix', '?')) + '\n')
    f.write('//nobj: ' + str(_arg('nobj', '?')) + '\n\n')

    nobj_avg = _arg('nobj_avg', 49)
    f.write('//normalization constants\n')
    f.write('//nobj avg = {}\n'.format(nobj_avg))
    f.write('internal_t invnave = {};\n'.format(1 / nobj_avg))
    f.write('internal_t invnave2 = {};\n\n'.format(1 / nobj_avg ** 2))

    f.write('//first batchnorm [mean, weight/sqrt(var), bias]\n')
    f.write('weight_t batch1_2to2[3] = ' + _c(batch1) + ';\n\n')

    f.write('//2to2 linear layer\n')
    f.write(f'{w1_type} w1_2to2[NHIDDEN*6] = ' + _c(w1_2to2) + ';\n')
    f.write('bias_t b1_2to2[NHIDDEN] = ' + _c(b1_2to2) + ';\n')
    f.write('bias_t b1_diag_2to2[NHIDDEN] = ' + _c(b1d_2to2) + ';\n\n')

    f.write('//second batchnorm [channel][mean, weight/sqrt(var), bias]\n')
    f.write('weight_t batch2_2to0[NHIDDEN][3] = ' + _c(batch2) + ';\n\n')

    f.write('//2to1 linear layer\n')
    f.write(f'{w2_type} w2_2to0[NHIDDEN*2*NOUT] = ' + _c(w2_2to0) + ';\n')
    f.write('bias_t b2_2to0[NOUT] = ' + _c(b2_2to0) + ';\n')

print(f'\nWrote {args.out}  (NHIDDEN={NHIDDEN}, NOUT={NOUT})')
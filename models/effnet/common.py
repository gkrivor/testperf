import os
import sys

# EfficientNet version -> input image size (square). See:
# https://huggingface.co/google/efficientnet-b0 ... b7
EFFICIENTNET_IMAGE_SIZES = {
    'b0': 224,
    'b1': 240,
    'b2': 260,
    'b3': 300,
    'b4': 380,
    'b5': 456,
    'b6': 528,
    'b7': 600,
}

# Reverse lookup: image size -> version. All sizes are unique.
SIZE_TO_VERSION = {size: version for version, size in EFFICIENTNET_IMAGE_SIZES.items()}

DEFAULT_VERSION = 'b3'


def _valid_options_message():
  versions = ', '.join(f'{v} ({s})' for v, s in EFFICIENTNET_IMAGE_SIZES.items())
  return f'Valid EfficientNet versions/sizes: {versions}'


def get_image_size(version):
  """Return the square input image size for an EfficientNet version (e.g. 'b3' -> 300)."""
  version = str(version).lower()
  if version not in EFFICIENTNET_IMAGE_SIZES:
    raise Exception(f'Unknown EfficientNet version "{version}". {_valid_options_message()}')
  return EFFICIENTNET_IMAGE_SIZES[version]


def resolve_version(spec):
  """Normalize a single --imgsz value into a 'bX' version.

  Accepts:
    - a version 'bX' (case-insensitive), e.g. 'b3'
    - a single dimension, e.g. '300'
    - both dimensions, e.g. '300x300', '300,300' or '300 300'
  Numeric specs are resolved back to a version via the image-size table. A spec
  that does not correspond to an existing model raises a clear error.
  """
  if spec is None:
    raise Exception(f'Missing --imgsz value. {_valid_options_message()}')
  spec = str(spec).strip().lower()
  if not spec:
    raise Exception(f'Empty --imgsz value. {_valid_options_message()}')

  # Version form: bX
  if spec.startswith('b'):
    if spec not in EFFICIENTNET_IMAGE_SIZES:
      raise Exception(f'Unknown EfficientNet version "{spec}". {_valid_options_message()}')
    return spec

  # Numeric form: single or both dimensions (separated by 'x', ',' or whitespace)
  parts = spec.replace('x', ' ').replace(',', ' ').split()
  try:
    dims = [int(p) for p in parts]
  except ValueError:
    raise Exception(f'Invalid --imgsz value "{spec}". {_valid_options_message()}')

  if len(dims) == 0:
    raise Exception(f'Invalid --imgsz value "{spec}". {_valid_options_message()}')
  if len(dims) > 2:
    raise Exception(f'Invalid --imgsz value "{spec}", expected 1 or 2 dimensions. {_valid_options_message()}')
  if len(dims) == 2 and dims[0] != dims[1]:
    raise Exception(
        f'EfficientNet inputs are square; --imgsz dimensions must be equal, got {dims[0]}x{dims[1]}.'
    )

  size = dims[0]
  if size not in SIZE_TO_VERSION:
    raise Exception(f'No EfficientNet model with image size {size}. {_valid_options_message()}')
  return SIZE_TO_VERSION[size]


def get_efficientnet_version(argv=None):
  """Resolve the EfficientNet version from the --imgsz argument (default b3)."""
  argv = sys.argv if argv is None else argv
  if '--imgsz' in argv:
    try:
      spec = argv[argv.index('--imgsz') + 1]
    except IndexError:
      raise Exception(f'Missing value after --imgsz. {_valid_options_message()}')
    return resolve_version(spec)
  return DEFAULT_VERSION


def is_fp16(argv=None):
  """Return True when --fp16 is present in the command line."""
  argv = sys.argv if argv is None else argv
  return '--fp16' in argv


def onnx_name(version, batch, half):
  """Build the ONNX file name for a version/batch/precision combination."""
  precision = 'fp16' if half else 'fp32'
  return f'efficientnet-{str(version).lower()}_{precision}_{batch}b.onnx'


def get_input_np_dtype(half):
  """Numpy dtype for the model input given the precision flag."""
  import numpy as np
  return np.float16 if half else np.float32

def try_export_model(file_path, batch_size, version, half_precision=False, opset=17):
  """Export google/efficientnet-<version> to ONNX (ported from effconv.py).

  The exported graph takes a preprocessed ``pixel_values`` tensor of shape
  (batch, 3, size, size) and returns the raw ``logits`` tensor.
  """
  if os.path.exists(file_path):
    return

  try:
    import torch
    from transformers import EfficientNetForImageClassification
  except ImportError as exc:
    raise Exception(
        'Missing dependencies for EfficientNet export. Install them in the venv, e.g.:\n'
        '    python -m pip install torch transformers\n'
        f'Original import error: {exc}'
    )

  version = str(version).lower()
  model_id = f'google/efficientnet-{version}'
  size = get_image_size(version)

  model = None
  try:
    model = EfficientNetForImageClassification.from_pretrained(model_id)
  except Exception as e:
    raise Exception('Next dependencies might be required:\n    python -m pip install onnxscripts pip-system-certs\n'
    f'Original error: {e}')
  model.eval()

  class LogitsWrapper(torch.nn.Module):
    """Return the raw logits tensor instead of an ImageClassifierOutput."""

    def __init__(self, wrapped):
      super().__init__()
      self.model = wrapped

    def forward(self, pixel_values):
      return self.model(pixel_values=pixel_values).logits

  wrapped = LogitsWrapper(model)

  # FP16 export uses CUDA when available; some ops are unsupported in half on CPU.
  device = 'cpu'
  if half_precision and torch.cuda.is_available():
    device = 'cuda'

  wrapped = wrapped.to(device)
  dummy_input = torch.randn(batch_size, 3, size, size, dtype=torch.float32, device=device)
  if half_precision:
    wrapped = wrapped.half()
    dummy_input = dummy_input.half()

  with torch.no_grad():
    torch.onnx.export(
        wrapped,
        dummy_input,
        file_path,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

  precision = 'fp16' if half_precision else 'fp32'
  print(f'Exported {model_id} to {file_path} (opset {opset}, batch {batch_size}, {precision}).')

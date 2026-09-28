import os
import re
import sys

import numpy as np

# Mapping from ONNX Runtime tensor type strings to numpy dtypes
ORT_TO_NP = {
  'tensor(float)': np.float32,
  'tensor(float16)': np.float16,
  'tensor(double)': np.float64,
  'tensor(bfloat16)': np.float32,  # numpy has no bfloat16; fall back to float32
  'tensor(int64)': np.int64,
  'tensor(int32)': np.int32,
  'tensor(int16)': np.int16,
  'tensor(int8)': np.int8,
  'tensor(uint64)': np.uint64,
  'tensor(uint32)': np.uint32,
  'tensor(uint16)': np.uint16,
  'tensor(uint8)': np.uint8,
  'tensor(bool)': np.bool_,
}


def get_model_path(args=None):
  """Return the ONNX model path passed via --model "path/to/model.onnx".

  Raises a clear exception when the flag is missing or the file does not exist.
  The path is returned as-is (absolute or relative to the current directory).
  """
  if args is None:
    args = sys.argv
  if '--model' not in args:
    raise Exception('Missing --model "path\\to\\model.onnx"')
  try:
    model_path = args[args.index('--model') + 1]
  except IndexError:
    raise Exception('Missing value for --model "path\\to\\model.onnx"')
  if not model_path or model_path.startswith('--'):
    raise Exception('Missing value for --model "path\\to\\model.onnx"')
  if not os.path.exists(model_path):
    raise Exception(f'Model file not found: {model_path}')
  return model_path


def _parse_shape_dims(flag, dims_str):
  """Parse the comma-separated dims inside a name[...] override into a list.

  Each (stripped) token that is '?' (dynamic) or empty maps to None; any other
  token must be an integer. None dims flow through resolve_shape's dynamic
  handling (leading dynamic -> batch_size, others -> 1).
  """
  if not dims_str.strip():
    return []
  dims = []
  for token in dims_str.split(','):
    token = token.strip()
    if token == '' or token == '?':
      dims.append(None)
      continue
    try:
      dims.append(int(token))
    except ValueError:
      raise Exception(
        f'Invalid dimension {token!r} in {flag} shape; expected an integer or "?"')
  return dims


def parse_shape_args(flag, ext_params=None):
  """Parse shape-override arguments of the form --flag "name[shape]anydata,...".

  Reads sys.argv directly. Accepts multiple occurrences of ``flag`` (each value
  accumulates) and multiple definitions per value. Definitions are delimited by
  the sequence ",name[" (a comma immediately followed by a new name[ group), so
  commas inside a shape or inside the trailing "anydata" are preserved.

  A single definition has the form ``name[dims]anydata`` where the trailing
  ``anydata`` (any text after ']', may contain commas, assumed not to contain
  '['/']') is optional. Whitespace (spaces, tabs, newlines, carriage returns)
  around names, dims, and anydata is ignored.

  Returns a dict mapping ``name`` -> list of dims (ints, or None for '?'/empty).
  Later definitions for the same name override earlier ones.

  If a mutable ``ext_params`` dict is supplied, it is filled with the raw
  trailing data: ``ext_params[name] = anydata`` (stripped string, or None when
  absent). The anydata is otherwise ignored. It is reserved for a future use
  case -- e.g. ``name[shape]=src/file/path`` to read/write/verify tensor data
  from an external file instead of random data. The interpretation of anydata
  (such as a leading '=') is intentionally left to that future consumer.
  """
  overrides = {}
  argv = sys.argv
  for index, token in enumerate(argv):
    if token != flag:
      continue
    try:
      value = argv[index + 1]
    except IndexError:
      raise Exception(f'Missing value for {flag} "name[shape]"')
    if not value or value.startswith('--'):
      raise Exception(f'Missing value for {flag} "name[shape]"')
    # Split into definition chunks only at commas that precede a new name[ group
    for chunk in re.split(r',(?=\s*[^\[\],]+\[)', value):
      if not chunk.strip():
        continue
      match = re.match(r'^\s*([^\[\]]+)\[([^\]]*)\]\s*(.*)$', chunk, re.DOTALL)
      if not match:
        raise Exception(
          f'Invalid {flag} definition {chunk.strip()!r}; expected name[shape]')
      name = match.group(1).strip()
      if not name:
        raise Exception(
          f'Invalid {flag} definition {chunk.strip()!r}; missing name')
      overrides[name] = _parse_shape_dims(flag, match.group(2))
      if ext_params is not None:
        anydata = match.group(3).strip()
        ext_params[name] = anydata if anydata else None
  return overrides


def parse_input_shapes(ext_params=None):
  """Parse --input "name[shape]..." overrides from sys.argv (see parse_shape_args)."""
  return parse_shape_args('--input', ext_params)


def parse_output_shapes(ext_params=None):
  """Parse --output "name[shape]..." overrides from sys.argv (see parse_shape_args).

  Symmetrical to parse_input_shapes; not consumed by any provider yet but kept
  parallel to inputs for future use.
  """
  return parse_shape_args('--output', ext_params)


def _apply_shape_overrides(descriptors, shape_overrides, kind):
  """Substitute descriptor shapes with user-supplied overrides (in place).

  Raises when an override name does not match any model descriptor (typo
  protection). ``kind`` is 'input' or 'output' for clear error messages.
  """
  if not shape_overrides:
    return
  names = {descriptor['name'] for descriptor in descriptors}
  for name in shape_overrides:
    if name not in names:
      raise Exception(f'{kind.capitalize()} name not found in model: {name}')
  for descriptor in descriptors:
    if descriptor['name'] in shape_overrides:
      descriptor['shape'] = list(shape_overrides[descriptor['name']])


def get_model_inputs(sess, shape_overrides=None):
  """Return the model input descriptors as a list of {name, shape, type}.

  When ``shape_overrides`` is None, command-line --input overrides are parsed
  from sys.argv; pass an explicit dict (name -> dim list) to override that.
  """
  if shape_overrides is None:
    shape_overrides = parse_input_shapes()
  descriptors = [{'name': node.name, 'shape': list(node.shape), 'type': node.type}
                 for node in sess.get_inputs()]
  _apply_shape_overrides(descriptors, shape_overrides, 'input')
  return descriptors


def get_model_outputs(sess, shape_overrides=None):
  """Return the model output descriptors as a list of {name, shape, type}.

  When ``shape_overrides`` is None, command-line --output overrides are parsed
  from sys.argv; pass an explicit dict (name -> dim list) to override that.
  """
  if shape_overrides is None:
    shape_overrides = parse_output_shapes()
  descriptors = [{'name': node.name, 'shape': list(node.shape), 'type': node.type}
                 for node in sess.get_outputs()]
  _apply_shape_overrides(descriptors, shape_overrides, 'output')
  return descriptors


def get_ort_input_np_dtype(node_or_type):
  """Map an ONNX Runtime input (NodeArg or type string) to a numpy dtype."""
  type_str = node_or_type if isinstance(node_or_type, str) else node_or_type.type
  if type_str not in ORT_TO_NP:
    raise Exception(f'Unsupported ONNX Runtime input type: {type_str}')
  return ORT_TO_NP[type_str]


def resolve_shape(shape, batch_size):
  """Resolve a (possibly dynamic) ONNX shape into a concrete tuple of ints.

  The leading dynamic dimension (a symbolic string or None) is replaced with
  batch_size; any other unknown dimension is replaced with 1. Static integer
  dimensions are kept unchanged.
  """
  resolved = []
  leading_dynamic_used = False
  for index, dim in enumerate(shape):
    if isinstance(dim, int) and dim > 0:
      resolved.append(dim)
      continue
    # Dynamic / symbolic / unknown dimension
    if index == 0 and not leading_dynamic_used:
      resolved.append(int(batch_size))
      leading_dynamic_used = True
    else:
      resolved.append(1)
  return tuple(resolved)


def _random_array(shape, dtype):
  """Create a random numpy array of the given shape and dtype."""
  np_dtype = np.dtype(dtype)
  if np_dtype == np.bool_:
    return np.random.randint(0, 2, size=shape).astype(np.bool_)
  if np.issubdtype(np_dtype, np.integer):
    info = np.iinfo(np_dtype)
    low = max(int(info.min), 0)
    high = min(int(info.max), 100) + 1
    return np.random.randint(low, high, size=shape).astype(np_dtype)
  # Floating point types
  return np.random.randn(*shape).astype(np_dtype) if shape else np.random.randn(1).astype(np_dtype).reshape(())


def random_input_feed(sess, batch_size, shape_overrides=None):
  """Build a {name: np.ndarray} feed dict for every model input.

  Shapes are resolved via resolve_shape (after applying any --input command-line
  overrides) and dtypes via the model's declared input types, so this works for
  an arbitrary ONNX model. When ``shape_overrides`` is None the overrides are
  parsed from sys.argv; pass an explicit dict (name -> dim list) to override.
  """
  feed = {}
  for node in get_model_inputs(sess, shape_overrides):
    shape = resolve_shape(node['shape'], batch_size)
    dtype = get_ort_input_np_dtype(node['type'])
    feed[node['name']] = _random_array(shape, dtype)
  return feed

import os
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


def get_model_inputs(sess):
  """Return the model input descriptors as a list of {name, shape, type}."""
  return [{'name': node.name, 'shape': list(node.shape), 'type': node.type}
          for node in sess.get_inputs()]


def get_model_outputs(sess):
  """Return the model output descriptors as a list of {name, shape, type}."""
  return [{'name': node.name, 'shape': list(node.shape), 'type': node.type}
          for node in sess.get_outputs()]


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


def random_input_feed(sess, batch_size):
  """Build a {name: np.ndarray} feed dict for every model input.

  Shapes are resolved via resolve_shape and dtypes via the model's declared
  input types, so this works for an arbitrary ONNX model.
  """
  feed = {}
  for node in sess.get_inputs():
    shape = resolve_shape(list(node.shape), batch_size)
    dtype = get_ort_input_np_dtype(node.type)
    feed[node.name] = _random_array(shape, dtype)
  return feed

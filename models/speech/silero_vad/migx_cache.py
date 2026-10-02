import os

import numpy as np
import migraphx

from .common import CONTEXT, SAMPLE_RATE, STATE_SIZE, WINDOW, SileroNumpyStream


class Model(SileroNumpyStream):
  """Silero VAD streaming inference using direct MIGraphX with cache.

  The stock silero_vad.onnx (If/LSTM graph) does not compile with MIGraphX. Pass an
  If-free export with --model; MIGRAPHX_DISABLE_PASSES (e.g. gpu::fuse_mlir) is taken
  from the environment and recorded in the report.
  """
  def __init__(self):
    super().__init__('silero_vad.onnx')
    self.program = None
    self.compiled_batches = []

  def _cache_path(self, batch_size):
    base = os.path.splitext(os.path.basename(self.model_file))[0]
    return self.get_file_path(f'{base}_b{batch_size}.mxr')

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['MIGRAPHX_DISABLE_PASSES'] = os.environ.get('MIGRAPHX_DISABLE_PASSES', '')
    self.compiled_batches.append(batch_size)
    cache_path = self._cache_path(batch_size)
    if os.path.exists(cache_path):
      return
    try:
      program = migraphx.parse_onnx(self.model_file, map_input_dims={
        'input': [batch_size, CONTEXT + WINDOW],
        'state': [2, batch_size, STATE_SIZE],
      })
      program.compile(migraphx.get_target('gpu'))
      migraphx.save(program, cache_path)
    except Exception as e:
      raise Exception(f'Failed to compile {self.model_file} with MIGraphX (the stock If/LSTM graph is not supported): {e}')

  def read(self):
    # Read timings run before any batch is selected, so they use the first compiled batch
    batch_size = self.batch_size if self.batch_size in self.compiled_batches else self.compiled_batches[0]
    self.program = migraphx.load(self._cache_path(batch_size))
    shapes = self.program.get_parameter_shapes()
    self.sr = migraphx.fill_argument(shapes['sr'], SAMPLE_RATE) if 'sr' in shapes else None

  def run_chunk(self, step):
    window = self.window(step)
    params = {'input': migraphx.argument(window), 'state': migraphx.argument(self.state)}
    if self.sr is not None:
      params['sr'] = self.sr
    outputs = self.program.run(params)
    self.state = np.array(outputs[1], copy=True)
    self.context = window[:, -CONTEXT:]
    return np.array(outputs[0], copy=True)

  def shutdown(self):
    self.program = None

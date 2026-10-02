import torch

from ..common import get_arg
from .torch_common import SileroRebuiltStream


class Model(SileroRebuiltStream):
  """Silero VAD streaming inference using the rebuilt nn.Module with torch.compile.

  --compile-mode reduce-overhead|default|max-autotune (default reduce-overhead, i.e. CUDA/HIP graphs)
  --compile-backend NAME   (default inductor)
  """
  def __init__(self):
    super().__init__()
    self.compile_mode = get_arg('--compile-mode', 'reduce-overhead')
    self.compile_backend = get_arg('--compile-backend', 'inductor')
    self.uses_graphs = self.compile_mode == 'reduce-overhead' and self.device.type == 'cuda' and self.compile_backend == 'inductor'
    # Graph capture happens on the first calls after compilation
    self.empty_runs = 3

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Compile Mode / Backend'] = f'{self.compile_mode} / {self.compile_backend}'

  def make_step(self, model):
    # Only inductor accepts a mode
    mode = self.compile_mode if self.compile_backend == 'inductor' else None
    return torch.compile(super().make_step(model), backend=self.compile_backend, mode=mode, dynamic=False, fullgraph=True)

  def forward_chunk(self, chunk):
    if self.uses_graphs:
      torch.compiler.cudagraph_mark_step_begin()
    prob, context, state = self.step_fn(chunk, self.context, self.state)
    if self.uses_graphs:
      # Graph outputs are overwritten by the next replay
      context, state = context.clone(), state.clone()
    self.context, self.state = context, state
    return prob

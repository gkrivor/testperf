import torch

from ..common import SpeechModel, get_arg, torch_device, torch_sync
from .common import designed_wait_ms, selected_geometries


class Model(SpeechModel):
  """Streaming Sortformer diarization inference with using NeMo (PyTorch).

  Batch N diarizes N files of similar length per diarize() call.
    --model ID|PATH.nemo   (default nvidia/diar_streaming_sortformer_4spk-v2.1)
    --geometry NAME        one geometry per run (default card_1.04s), see sortformer/common.py
  Accuracy (DER) is not scored: LibriSpeech is single-speaker. Segment counts are reported.
  """
  def __init__(self):
    super().__init__()
    self.device = torch_device()
    self.model_id = get_arg('--model', 'nvidia/diar_streaming_sortformer_4spk-v2.1')
    geometries = selected_geometries(['card_1.04s'])
    if len(geometries) != 1:
      raise Exception('Pass exactly one --geometry, run the test again for another one')
    self.geometry_name, self.geometry = geometries[0]
    self.model = None
    self.segments = {}

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Model'] = self.model_id
    self.details['Device'] = str(self.device)
    self.details['Geometry'] = f'{self.geometry_name} ' + ' '.join(f'{k}={v}' for k, v in self.geometry.items())
    self.details['Designed Wait (ms)'] = designed_wait_ms(self.geometry)

  def read(self):
    from nemo.collections.asr.models import SortformerEncLabelModel
    if self.model_id.endswith('.nemo'):
      model = SortformerEncLabelModel.restore_from(self.model_id, map_location=self.device)
    else:
      model = SortformerEncLabelModel.from_pretrained(self.model_id, map_location=self.device)
    self.model = model.to(self.device).eval()
    modules = self.model.sortformer_modules
    modules.chunk_len = self.geometry['chunk']
    modules.chunk_right_context = self.geometry['rc']
    modules.fifo_len = self.geometry['fifo']
    modules.spkcache_len = self.geometry['spkcache']
    modules.spkcache_update_period = self.geometry['update-period']
    modules._check_streaming_parameters()

  def inference(self):
    paths = [str(self.corpus.files[i]) for i in self.current_unit()]
    with torch.inference_mode():
      segments = self.model.diarize(audio=paths, batch_size=len(paths))
      torch_sync(self.device)
    self.segments[self.current_unit_index()] = sum(len(s) for s in segments) if segments else 0
    return segments

  def shutdown(self):
    if self.segments:
      self.details[f'Segments (batch {self.batch_size}, {len(self.segments)} groups)'] = sum(self.segments.values())
      self.segments = {}
    if self.model is not None:
      del self.model
      self.model = None
      if self.device.type == 'cuda':
        torch.cuda.empty_cache()

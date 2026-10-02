import torch

from ..common import SpeechModel, get_arg, torch_device, torch_dtype, torch_sync

KMEANS_URI = 'https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy'
STAGES = ('full', 'units', 'aligner')


class Model(SpeechModel):
  """SeamlessM4T UnitY2 speech-text alignment inference with using fairseq2 (seamless_communication).

  Needs reference transcripts (refs.tsv, see --refs); files without one are skipped.
  Batch N aligns N files sequentially per run (the extractor API is per utterance).
    --stage full|units|aligner   extract_alignment, unit extraction only, or aligner given
                                 precomputed units (default full)
    --dtype fp16|bf16|fp32       (default fp16)
  Requires fairseq2 0.2 and seamless_communication, see dockers/speech/Dockerfile_unity2.
  """
  def __init__(self):
    super().__init__()
    self.device = torch_device()
    self.dtype = torch_dtype('fp16')
    self.stage = get_arg('--stage', 'full')
    if self.stage not in STAGES:
      raise Exception(f'Unsupported --stage {self.stage}, use one of {", ".join(STAGES)}')
    self.extractor = None
    self.waves = {}
    self.units_cache = {}

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Stage'] = self.stage
    self.details['Device / DType'] = f'{self.device} / {self.dtype}'
    self.details['Aligner'] = 'nar_t2u_aligner + xlsr2_1b_v2 layer 35 + kmeans_10k'

  def read(self):
    from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
    self.extractor = AlignmentExtractor(
      aligner_model_name_or_card='nar_t2u_aligner',
      unit_extractor_model_name_or_card='xlsr2_1b_v2',
      unit_extractor_output_layer=35,
      unit_extractor_kmeans_model_uri=KMEANS_URI,
      device=self.device,
      dtype=self.dtype,
    )

  def make_units(self, batch_size):
    with_refs = [i for i in range(len(self.corpus)) if self.corpus.reference(i)]
    if not with_refs:
      raise Exception('No reference transcripts: pass --refs FILE or use --download')
    self.details['Files With References'] = len(with_refs)
    return self.corpus.groups(batch_size, with_refs)

  def prepare(self):
    super().prepare()
    with torch.inference_mode():
      for unit in self.get_units():
        for index in unit:
          if index not in self.waves:
            self.waves[index] = torch.from_numpy(self.corpus.audio(index))
          if self.stage == 'aligner' and index not in self.units_cache:
            self.units_cache[index] = self.extractor.extract_units(self.extractor.prepare_audio(self.waves[index]))

  def inference(self):
    with torch.inference_mode():
      for index in self.current_unit():
        if self.stage == 'units':
          self.extractor.extract_units(self.extractor.prepare_audio(self.waves[index]))
          continue
        source = self.units_cache[index] if self.stage == 'aligner' else self.waves[index]
        self.extractor.extract_alignment(source, self.corpus.reference(index), plot=False, add_trailing_silence=False)
      torch_sync(self.device)

  def shutdown(self):
    if self.extractor is not None:
      del self.extractor
      self.extractor = None
      self.units_cache = {}
      if self.device.type == 'cuda':
        torch.cuda.empty_cache()

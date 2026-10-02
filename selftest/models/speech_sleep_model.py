from time import sleep

from models.speech.common import SpeechModel


class Model(SpeechModel):
  """Speech model for testing: sleeps 2 ms per group of files"""
  def inference(self):
    sleep(0.002)
    return self.current_unit()

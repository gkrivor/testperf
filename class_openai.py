import os
from pathlib import Path
from class_model import Model

class OpenAIModel(Model):
    def __init__(self):
        super().__init__()
        self.config = {}

from models.glm.glm_52.vllm_cli import Model as BaseModel
import itertools
import sys
import test_perf

class Model(BaseModel):
    """GLM-5.2-MXFP4 inference with using VLLM CLI"""
    def __init__(self):
        super().__init__()
        # Info for generating batches
        context_length = [8192]
        random_input_length = [1024, 4096]
        random_output_length = [1024, 4096]
        # Generate batches
        self.batches = list(itertools.product(context_length, random_input_length, random_output_length))

        # Need prepare batch only once
        self.is_batch_prepared = False

        # Overriding batch list in case it isn't explicitly set
        if sys.argv.count('--batch-size') == 0:
            test_perf.batches = [x for x in range(len(self.batches))]

    def prepare_batch(self, batch_size):
        if self.is_batch_prepared:
            return
        self.vllm_config['max_model_len'] = self.batches[batch_size][0]
        super().prepare_batch(batch_size)
        self.is_batch_prepared = True

    def prepare(self):
        # Here we can set settings for the selected batch
        batch = self.batches[self.batch_size]
        if self.vllm_config['max_model_len'] != batch[0]:
            self.vllm_config['max_model_len'] = batch[0]
            # Server will be started again while nearest read() call
            self._stop_server()
        self.bench_config['random-input-len'] = batch[1]
        self.bench_config['random-output-len'] = batch[2]
        super().prepare()
from class_vllm import VLLMBench as BaseModel

class Model(BaseModel):
    """Qwen3-8B inference with using VLLM CLI"""
    def __init__(self):
        super().__init__(model_id='Qwen/Qwen3-8B')
        # VLLM Server-specific settings
        self.vllm_config['max_model_len'] = 4096
        # VLLM Bench-specific settings
        self.bench_config['num-prompts'] = 500
        # Test-specific settings
        self.random_output_length = 128

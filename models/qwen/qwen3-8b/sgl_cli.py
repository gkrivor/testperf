from class_sglang import SGLangBench as BaseModel

class Model(BaseModel):
    """Qwen3-8B inference with using SGLang CLI"""
    def __init__(self):
        super().__init__(model_id='Qwen/Qwen3-8B')
        # SGLang Server-specific settings
        self.sglang_config['context-length'] = 4096
        # SGLang Bench-specific settings
        self.bench_config['num-prompts'] = 500
        # Test-specific settings
        self.random_output_length = 128

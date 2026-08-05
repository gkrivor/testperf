from class_sglang import SGLangOpenAI as BaseModel

class Model(BaseModel):
    """Qwen3-8B inference with using default SGLang OpenAI API"""
    def __init__(self):
        super().__init__(model_id='Qwen/Qwen3-8B')
        # SGLang Server-specific settings
        self.sglang_config['context-length'] = 4096
        # Test-specific settings
        self.random_input_length = 2048

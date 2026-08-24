from class_vllm import VLLMOpenAI as BaseModel

class Model(BaseModel):
    """Qwen3-8B inference with using default VLLM OpenAI API"""
    def __init__(self):
        super().__init__(model_id='Qwen/Qwen3-8B')
        # VLLM Server-specific settings
        self.vllm_config['max_model_len'] = 4096
        # Test-specific settings
        self.random_input_length = 2048

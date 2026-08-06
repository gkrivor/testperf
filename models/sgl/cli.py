import sys
from class_sglang import SGLangBench as BaseModel

class Model(BaseModel):
    """Custom model inference with using SGLang CLI"""
    def __init__(self):
        model_id = None
        try:
            model_id = sys.argv[sys.argv.index('--model') + 1]
        except IndexError:
            print('Error: Model ID is required, use --model <model_id> to specify the model')
            sys.exit(1)
        # Update the docstring to use in reports later
        self.__doc__ = f'Custom {model_id} inference with using SGLang CLI'
        super().__init__(model_id=model_id)
        # Test-specific settings
        self.random_output_length = 128

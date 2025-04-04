import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel=False):
        self.model_name = model_name
        self.config = config
        self.use_accel = use_accel
        self._model = None

    def _lazy_import(self, module_name, func_name):
        if module_name.startswith('.'):
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def load_model(self):
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if 'api' in self.config.get('call_type'):
                self._model = load_func(
                    self.config['model_path_or_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model'],
                    self.config['call_type']
                )
            else:
                self._model = load_func(self.model_name, self.config, self.use_accel)
        return self._model

    @property
    def model(self):
        return self.load_model()

    @property
    def infer(self):
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel=False)

    def load_model(self, choice, use_accel=False):
        """Load a model based on the choice."""
        if choice in self.models:
            self.models[choice].use_accel = use_accel
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice):
        """Get the inference function for a given model."""
        if choice in self.models:
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    ####### APi models #######
    'gpt-4o': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4o',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4o-2024-05-13',
        'call_type': 'api_chat'
    },
    'gpt-4o-2024-08-06': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4o',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4o-2024-08-06',
        'call_type': 'api_chat'
    },
    'gpt-4-turbo-2024-04-09': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4-turbo',
        'call_type': 'api_chat'
    },
    'claude-3-5-sonnet-20240620': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Claude-3-5-Sonnet-20240620',
        'base_url': '',
        'api_key': '',
        'model': 'claude-3-5-sonnet-20240620',
        'call_type': 'api_chat'
    },
    'doubao-pro-128k': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Doubao-Pro',
        'base_url': "", 
        'api_key': "", 
        'model': "",
        'call_type': 'api_chat'
    },

    ####### Local Language Aligned models #######
    'Qwen2.5-0.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-1.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-7B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-7B-Instruct',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-14B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-14B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-32B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-32B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-72B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-72B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2-0.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-0.5B-Instruct',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2-1.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-1.5B-Instruct',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2-7B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-7B-Instruct',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2-72B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-72B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Phi-3.5-MoE-instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'microsoft/Phi-3.5-MoE-instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Phi-3.5-mini-instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'microsoft/Phi-3.5-mini-instruct',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-2b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-2b-it',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-9b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-9b-it',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-27b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-27b-it',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3-8B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3-70B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-8B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3.1-70B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-70B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-6B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-6B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-9B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-9B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-34B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-34B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'MAP-Neo-7B-Instruct-v0.1': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'm-a-p/neo_7b_instruct_v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-7B-Instruct-v0.3': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-Large-Instruct-2407': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/hf_models/Mistral-Large-Instruct-2407',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-7B-0724-Instruct-hf': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'allenai/OLMo-7B-0724-Instruct-hf',
        'call_type': 'local',
        'tp': 8
    },
    'c4ai-command-r-plus-08-2024': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'CohereForAI/c4ai-command-r-plus-08-2024',
        'call_type': 'local',
        'tp': 8
    },
    'c4ai-command-r-08-2024': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'CohereForAI/c4ai-command-r-08-2024',
        'call_type': 'local',
        'tp': 8
    },
    'DeepSeek-V2.5': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'deepseek-ai/DeepSeek-V2.5',
        'call_type': 'local',
        'tp': 8
    },

    ####### Local Language Base models #######
    'Yi-1.5-6B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-6B',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-9B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-9B',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-34B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-34B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-0.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-0.5B',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-1.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-1.5B',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-7B',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-14B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-14B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-32B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-32B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-72B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-72B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-7B',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2-72B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-72B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2-0.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-0.5B',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2-1.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2-1.5B',
        'call_type': 'local',
        'tp': 4
    },
    'gemma-2-2b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-2b',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-9b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-9b',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-27b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-27b',
        'call_type': 'local',
        'tp': 8
    },
    'MAP-Neo-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'm-a-p/neo_7b',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-7B-v0.1': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-7B-v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3-8B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Meta-Llama-3-8B',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3-70B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Meta-Llama-3-70B',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3.1-8B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-8B',
        'call_type': 'local',
        'tp': 8
    },
    'Meta-Llama-3.1-70B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-70B',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-7B-0724-hf': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'allenai/OLMo-7B-0724-hf',
        'call_type': 'local',
        'tp': 8
    },
}

# # Register all models
# for model_name, config in model_configs.items():
#     model_registry.register_model(model_name, config)

def load_model(choice, use_accel=False):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice])
    return model_registry.load_model(choice, use_accel)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)
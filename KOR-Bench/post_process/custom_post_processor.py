from eval.eval_utils import evaluate_responses
from utils.common import read_yaml
from config.config_wrapper import config_wrapper, get_config_wrapper


class PostProcessorRegistry:
    """Registry for post-processors to allow dynamic registration and retrieval."""
    _registry = {}

    @classmethod
    def register_processor(cls, name):
        """Decorator to register a custom post-processor by name."""
        def wrapper(processor_cls):
            if name in cls._registry:
                raise ValueError(f"Processor '{name}' is already registered.")
            cls._registry[name] = processor_cls
            return processor_cls
        return wrapper

    @classmethod
    def get_processor(cls, name, *args, **kwargs):
        """Retrieve the post-processor by name."""
        if not isinstance(name, str):
            raise TypeError(f"Processor name must be a string, got {type(name)}: {name}")
        
        processor_cls = cls._registry.get(name)
        if not processor_cls:
            return None
        return processor_cls(*args, **kwargs)

    @classmethod
    def register_processors(cls, *names):
        """Decorator to register a custom post-processor by multiple names."""
        def wrapper(processor_cls):
            for name in names:
                if name in cls._registry:
                    raise ValueError(f"Processor '{name}' is already registered.")
                cls._registry[name] = processor_cls
            return processor_cls
        return wrapper


class BasePostProcessor:
    """Base class for post-processors. Custom processors should inherit from this."""
    def process(self, data):
        raise NotImplementedError("Subclasses must implement the 'process' method.")


@PostProcessorRegistry.register_processors("self-correction", "self-correction-with-needle")
class SelfCorrectionProcessor(BasePostProcessor):
    def process(self, samples):
        config_wrapper = get_config_wrapper()
        question_type = config_wrapper.split
        mode = config_wrapper.mode
        max_rounds = config_wrapper.max_rounds
        evaluation_results = evaluate_responses(samples, question_type, mode)
        results_to_save = []
        results_to_return = []
        for sample, result in zip(samples, evaluation_results):
            if result['is_correct']:
                sample[config_wrapper.status_key] = 'correct'
                results_to_save.append(sample)
            elif result['is_correct'] == False and isinstance(sample[config_wrapper.response_key], dict) and config_wrapper.error_key in sample[config_wrapper.response_key]:
                sample[config_wrapper.status_key] = 'error'
                results_to_save.append(sample)
            else:
                template = read_yaml('self-correction')
                sample[config_wrapper.status_key] = 'processing'
                history = sample.get('history', {})
                if len(history) + 1 >= max_rounds:
                    # sample[config_wrapper.history_key] = {}
                    sample[config_wrapper.status_key] = 'max_rounds'
                    results_to_save.append(sample)
                    continue
                history[len(history)] = {'prompt': sample[config_wrapper.prompt_key], 'response': str(result['response_text']), 'response_origin': sample[config_wrapper.response_key]}
                sample[config_wrapper.history_key] = history.copy()
                if mode == 'self-correction-with-needle':
                    needle_str = '\n'.join(sample['needle'])
                    sample[config_wrapper.prompt_key] = needle_str + '\n' + template['prompt_format'][0]
                else:
                    sample[config_wrapper.prompt_key] = template['prompt_format'][0]
                results_to_return.append(sample)
                # results_to_save.append(sample)

        return results_to_save, results_to_return

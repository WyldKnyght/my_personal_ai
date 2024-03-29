from ctransformers import AutoConfig, AutoModelForCausalLM

from user_interface import ui_settings as settings
from model_handler.callbacks import Iteratorize
from common.logging_colors import logger


class CtransformersModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path):
        result = cls()

        config = AutoConfig.from_pretrained(
            str(path),
            threads=settings.args.threads if settings.args.threads != 0 else -1,
            gpu_layers=settings.args.n_gpu_layers,
            batch_size=settings.args.n_batch,
            context_length=settings.args.n_ctx,
            stream=True,
            mmap=not settings.args.no_mmap,
            mlock=settings.args.mlock
        )

        result.model = AutoModelForCausalLM.from_pretrained(
            str(result.model_dir(path) if result.model_type_is_auto() else path),
            model_type=(None if result.model_type_is_auto() else settings.args.model_type),
            config=config
        )

        logger.info(f'Using ctransformers model_type: {result.model.model_type} for {result.model.model_path}')
        return result, result

    def model_type_is_auto(self):
        return settings.args.model_type is None or settings.args.model_type == "Auto" or settings.args.model_type == "None"

    def model_dir(self, path):
        if path.is_file():
            return path.parent

        return path

    def encode(self, string, **kwargs):
        return self.model.tokenize(string)

    def decode(self, ids):
        return self.model.detokenize(ids)

    def generate(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        # ctransformers uses -1 for random seed
        generator = self.model(
            prompt=prompt,
            max_new_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            repetition_penalty=state['repetition_penalty'],
            last_n_tokens=state['repetition_penalty_range'],
            seed=int(state['seed'])
        )

        output = ""
        for token in generator:
            if callback:
                callback(token)

            output += token

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

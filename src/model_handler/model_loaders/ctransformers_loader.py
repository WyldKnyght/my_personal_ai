# /src/model_handler/model_loaders/ctransformers_loader.py

from pathlib import Path
from user_interface import ui_settings as settings
from common.logging_colors import logger

def ctransformers_loader(model_name):
    from model_handler.ctransformers_model import CtransformersModel

    path = Path(f'{settings.args.model_dir}/{model_name}')
    ctrans = CtransformersModel()
    if ctrans.model_type_is_auto():
        model_file = path
    else:
        if path.is_file():
            model_file = path
        else:
            entries = Path(f'{settings.args.model_dir}/{model_name}')
            gguf = list(entries.glob('*.gguf'))
            bin = list(entries.glob('*.bin'))
            if len(gguf) > 0:
                model_file = gguf[0]
            elif len(bin) > 0:
                model_file = bin[0]
            else:
                logger.error("Could not find a model for ctransformers.")
                return None, None

    logger.info(f'ctransformers weights detected: {model_file}')
    model, tokenizer = ctrans.from_pretrained(model_file)
    return model, tokenizer


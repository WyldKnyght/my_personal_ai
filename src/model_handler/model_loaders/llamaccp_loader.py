# /src/model_handler/model_loaders/llamacpp_loader.py

from user_interface import ui_settings as settings
from common.logging_colors import logger
from pathlib import Path

def llamacpp_loader(model_name):
    from model_handler.llamacpp_model import LlamaCppModel

    path = Path(f'{settings.args.model_dir}/{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = list(Path(f'{settings.args.model_dir}/{model_name}').glob('*.gguf'))[0]

    logger.info(f"llama.cpp weights detected: {model_file}")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer
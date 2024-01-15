# /src/model_handler/model_loaders/llamacpp_HF_loader.py

from pathlib import Path
from user_interface import ui_settings as settings
from common.logging_colors import logger
from transformers import AutoTokenizer

def llamacpp_HF_loader(model_name):
    from model_handler.llamacpp_hf import LlamacppHF

    for fname in [model_name, "oobabooga_llama-tokenizer", "llama-tokenizer"]:
        path = Path(f'{settings.args.model_dir}/{fname}')
        if all((path / file).exists() for file in ['tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']):
            logger.info(f'Using tokenizer from: {path}')
            break
    else:
        logger.error("Could not load the model because a tokenizer in transformers format was not found. Please download oobabooga/llama-tokenizer.")
        return None, None

    if settings.args.no_use_fast:
        logger.info('Loading the tokenizer with use_fast=False.')

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=settings.args.trust_remote_code,
        use_fast=not settings.args.no_use_fast
    )

    model = LlamacppHF.from_pretrained(model_name)
    return model, tokenizer


# /src/model_handler/models.py

import gc
from pathlib import Path
import re
import time
import torch
from user_interface import ui_settings as settings
from accelerate.utils import is_xpu_available
from common.logging_colors import logger
from model_handler.models_settings import get_model_metadata
from model_handler.model_loaders.huggingface_loader import huggingface_loader
from model_handler.model_loaders.llamaccp_loader import llamacpp_loader
from model_handler.model_loaders.llamacpp_HF_loader import llamacpp_HF_loader
from model_handler.model_loaders.ctransformers_loader import ctransformers_loader
from transformers import AutoTokenizer

local_rank = None

def load_model(model_name, loader=None):
    logger.info(f"Loading {model_name}")
    t0 = time.time()

    settings.is_seq2seq = False
    settings.model_name = model_name
    load_func_map = {
        'Transformers': huggingface_loader,
        'llama.cpp': llamacpp_loader,
        'llamacpp_HF': llamacpp_HF_loader,
        'ctransformers': ctransformers_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if settings.args.loader is not None:
            loader = settings.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('The path to the model does not exist. Exiting.')
                raise ValueError

    settings.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    settings.settings.update({k: v for k, v in metadata.items() if k in settings.settings})
    if loader.lower().startswith('exllama'):
        settings.settings['truncation_length'] = settings.args.max_seq_len
    elif loader in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
        settings.settings['truncation_length'] = settings.args.n_ctx

    logger.info(f"LOADER: {loader}")
    logger.info(f"TRUNCATION LENGTH: {settings.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: {metadata['instruction_template']}")
    logger.info(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{settings.args.model_dir}/{model_name}/")
    if any(s in model_name.lower() for s in ['gpt-4chan', 'gpt4chan']) and Path(f"{settings.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{settings.args.model_dir}/gpt-j-6B/"))
    elif path_to_model.exists():
        if settings.args.no_use_fast:
            logger.info('Loading the tokenizer with use_fast=False.')

        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=settings.args.trust_remote_code,
            use_fast=not settings.args.no_use_fast
        )

    return tokenizer

def clear_torch_cache():
    gc.collect()
    if not settings.args.cpu:
        if is_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()

def unload_model():
    settings.model = settings.tokenizer = None
    settings.model_name = 'None'
    settings.lora_names = []
    settings.model_dirty_from_training = False
    clear_torch_cache()

def reload_model():
    unload_model()
    settings.model, settings.tokenizer = load_model(settings.model_name)

def update_truncation_length(current_length, state):
    # Check if 'loader' key is present in state dictionary
    if 'loader' in state:
        loader = state['loader'].lower()
        
        # Check if loader starts with 'exllama'
        if loader.startswith('exllama'):
            # Return 'max_seq_len' value
            return state['max_seq_len']
        
        # Check if loader matches specific values
        elif loader == 'llama.cpp' or loader == 'llamacpp_HF' or loader == 'ctransformers':
            # Return 'n_ctx' value
            return state['n_ctx']

    # Return current length if no conditions match
    return current_length

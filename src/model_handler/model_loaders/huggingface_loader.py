# /src/model_handler/huggingface_loader.py

from pathlib import Path
import traceback
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import is_xpu_available
from user_interface import ui_settings as settings
from common import RoPE
from common.logging_colors import logger
from model_handler.models_settings import get_max_memory_dict
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    GPTQConfig
)

def huggingface_loader(model_name):
    path_to_model = Path(f'{settings.args.model_dir}/{model_name}')
    params = {
        'low_cpu_mem_usage': True,
        'trust_remote_code': settings.args.trust_remote_code,
        'torch_dtype': torch.bfloat16 if settings.args.bf16 else torch.float16,
        'use_safetensors': True if settings.args.force_safetensors else None
    }

    if settings.args.use_flash_attention_2:
        params['use_flash_attention_2'] = True

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=params['trust_remote_code'])

    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            settings.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    # Load the model in simple 16-bit mode by default
    if not any([settings.args.cpu, settings.args.load_in_8bit, settings.args.load_in_4bit, settings.args.auto_devices, settings.args.disk, settings.args.deepspeed, settings.args.gpu_memory is not None, settings.args.cpu_memory is not None, settings.args.compress_pos_emb > 1, settings.args.alpha_value > 1, settings.args.disable_exllama, settings.args.disable_exllamav2]):
        model = LoaderClass.from_pretrained(path_to_model, **params)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            model = model.to(device)
        elif is_xpu_available():
            device = torch.device("xpu")
            model = model.to(device)
        else:
            model = model.cuda()

    # Load with quantization and/or offloading
    else:
        if not any((settings.args.cpu, torch.cuda.is_available(), is_xpu_available(), torch.backends.mps.is_available())):
            logger.warning('torch.cuda.is_available() and is_xpu_available() returned False. This means that no GPU has been detected. Falling back to CPU mode.')
            settings.args.cpu = True

        if settings.args.cpu:
            params['torch_dtype'] = torch.float32
        else:
            params['device_map'] = 'auto'
            params['max_memory'] = get_max_memory_dict()
            if settings.args.load_in_4bit:
                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval("torch.{}".format(settings.args.compute_dtype)) if settings.args.compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': settings.args.quant_type,
                    'bnb_4bit_use_double_quant': settings.args.use_double_quant,
                }

                logger.info('Using the following 4-bit params: ' + str(quantization_config_params))
                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif settings.args.load_in_8bit:
                if any((settings.args.auto_devices, settings.args.gpu_memory)):
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

                if params['max_memory'] is not None:
                    with init_empty_weights():
                        model = LoaderClass.from_config(config, trust_remote_code=params['trust_remote_code'])

                    model.tie_weights()
                    params['device_map'] = infer_auto_device_map(
                        model,
                        dtype=torch.int8,
                        max_memory=params['max_memory'],
                        no_split_module_classes=model._no_split_modules
                    )

            if settings.args.disk:
                params['offload_folder'] = settings.args.disk_cache_dir

        if settings.args.compress_pos_emb > 1:
            params['rope_scaling'] = {'type': 'linear', 'factor': settings.args.compress_pos_emb}
        elif settings.args.alpha_value > 1:
            params['rope_scaling'] = {'type': 'dynamic', 'factor': RoPE.get_alpha_value(settings.args.alpha_value, settings.args.rope_freq_base)}

        model = LoaderClass.from_pretrained(path_to_model, **params)

    return model
import functools
from collections import OrderedDict

import gradio as gr

from user_interface import ui_settings

loaders_and_params = OrderedDict({
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
        'load_in_8bit',
        'bf16',
        'cpu',
        'disk',
        'auto_devices',
        'load_in_4bit',
        'use_double_quant',
        'quant_type',
        'compute_dtype',
        'trust_remote_code',
        'no_use_fast',
        'use_flash_attention_2',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'disable_exllama',
        'disable_exllamav2',
        'transformers_info'
    ],
    'llama.cpp': [
        'n_ctx',
        'n_gpu_layers',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'no_offload_kqv',
        'tensorcores',
    ],
    'llamacpp_HF': [
        'n_ctx',
        'n_gpu_layers',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'cfg_cache',
        'trust_remote_code',
        'no_use_fast',
        'logits_all',
        'no_offload_kqv',
        'tensorcores',
        'llamacpp_HF_info',
    ],
    'ctransformers': [
        'n_ctx',
        'n_gpu_layers',
        'n_batch',
        'threads',
        'model_type',
        'no_mmap',
        'mlock'
    ],
})


def transformers_samplers():
    return {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'penalty_alpha',
        'num_beams',
        'length_penalty',
        'early_stopping',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    }


loaders_samplers = {
    'Transformers': transformers_samplers(),
    'llama.cpp': {
        'temperature',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'tfs',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'seed',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'ban_eos_token',
        'custom_token_bans',
    },
    'llamacpp_HF': {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'ctransformers': {
        'temperature',
        'top_p',
        'top_k',
        'repetition_penalty',
        'repetition_penalty_range',
    },
}

loaders_model_types = {
    'ctransformers': [
        "None",
        "llama",
        "dollyv2",
        "starcoder",
        "gptbigcode",
        "falcon"
    ],
}


@functools.cache
def list_all_samplers():
    all_samplers = set()
    for k in loaders_samplers:
        for sampler in loaders_samplers[k]:
            all_samplers.add(sampler)

    return sorted(all_samplers)


def blacklist_samplers(loader, dynamic_temperature):
    all_samplers = list_all_samplers()
    output = []

    for sampler in all_samplers:
        if loader == 'All' or sampler in loaders_samplers[loader]:
            if sampler.startswith('dynatemp'):
                output.append(gr.update(visible=dynamic_temperature))
            else:
                output.append(gr.update(visible=True))
        else:
            output.append(gr.update(visible=False))

    return output


def get_model_types(loader):
    if loader in loaders_model_types:
        return loaders_model_types[loader]

    return ["None"]


def get_gpu_memory_keys():
    return [k for k in ui_settings.gradio if k.startswith('gpu_memory')]


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    if 'gpu_memory' in all_params:
        all_params.remove('gpu_memory')
        for k in get_gpu_memory_keys():
            all_params.add(k)

    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

        if 'gpu_memory' in params:
            params.remove('gpu_memory')
            params += get_gpu_memory_keys()

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]

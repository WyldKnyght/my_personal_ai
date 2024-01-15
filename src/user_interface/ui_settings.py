# /src/user-interface/ui_settings.py

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

from common.logging_colors import logger

# UI Variables
gradio = {}
persistent_interface_state = {}
need_restart = False

# UI defaults
settings = {
    'dark_theme': True,
}

# Model variables
model = None
tokenizer = None
model_name = 'None'
is_seq2seq = False
model_dirty_from_training = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*Is typing...*'

# Parser copied from https://github.com/vladmandic/automatic
def parse_arguments():
    parser = argparse.ArgumentParser(description="Text generation web UI", conflict_handler='resolve', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))

    # Deprecated parameters
    args = parser.parse_args()
    args_defaults = parser.parse_args([])
    provided_arguments = []
    for arg in sys.argv[1:]:
        arg = arg.lstrip('-').replace('-', '_')
        if hasattr(args, arg):
            provided_arguments.append(arg)

    deprecated_args = []

    # Basic settings
    group = parser.add_argument_group('Basic settings')
    group.add_argument('--multi-user', action='store_true', help='Multi-user mode. Chat histories are not saved or automatically loaded. Warning: this is likely not safe for sharing publicly.')
    group.add_argument('--character', type=str, help='The name of the character to load in chat mode by default.')
    group.add_argument('--model', type=str, help='Name of the model to load by default.')
    group.add_argument('--lora', type=str, nargs='+', help='The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.')
    group.add_argument('--model-dir', type=str, default='models/', help='Path to directory with all the models.')
    group.add_argument('--lora-dir', type=str, default='loras/', help='Path to directory with all the loras.')
    group.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
    group.add_argument('--settings', type=str, help='Load the default interface settings from this yaml file. See settings-template.yaml for an example. If you create a file called settings.yaml, this file will be loaded by default without the need to use the --settings flag.')
    group.add_argument('--extensions', type=str, nargs='+', help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
    group.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
    group.add_argument('--chat-buttons', action='store_true', help='Show buttons on the chat tab instead of a hover menu.')

    # Transformers/Accelerate
    group = parser.add_argument_group('Transformers/Accelerate')
    group.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
    group.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
    group.add_argument('--gpu-memory', type=str, nargs='+', help='Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
    group.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
    group.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
    group.add_argument('--disk-cache-dir', type=str, default='cache', help='Directory to save the disk cache to. Defaults to "cache".')
    group.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision (using bitsandbytes).')
    group.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
    group.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.')
    group.add_argument('--trust-remote-code', action='store_true', help='Set trust_remote_code=True while loading the model. Necessary for some models.')
    group.add_argument('--force-safetensors', action='store_true', help='Set use_safetensors=True while loading the model. This prevents arbitrary code execution.')
    group.add_argument('--no_use_fast', action='store_true', help='Set use_fast=False while loading the tokenizer (it\'s True by default). Use this if you have any problems related to use_fast.')
    group.add_argument('--use_flash_attention_2', action='store_true', help='Set use_flash_attention_2=True while loading the model.')


    # bitsandbytes 4-bit
    group = parser.add_argument_group('bitsandbytes 4-bit')
    group.add_argument('--load-in-4bit', action='store_true', help='Load the model with 4-bit precision (using bitsandbytes).')
    group.add_argument('--use_double_quant', action='store_true', help='use_double_quant for 4-bit.')
    group.add_argument('--compute_dtype', type=str, default='float16', help='compute dtype for 4-bit. Valid options: bfloat16, float16, float32.')
    group.add_argument('--quant_type', type=str, default='nf4', help='quant_type for 4-bit. Valid options: nf4, fp4.')

    # llama.cpp
    group = parser.add_argument_group('llama.cpp')
    group.add_argument('--tensorcores', action='store_true', help='Use llama-cpp-python compiled with tensor cores support. This increases performance on RTX cards. NVIDIA only.')
    group.add_argument('--n_ctx', type=int, default=2048, help='Size of the prompt context.')
    group.add_argument('--threads', type=int, default=0, help='Number of threads to use.')
    group.add_argument('--threads-batch', type=int, default=0, help='Number of threads to use for batches/prompt processing.')
    group.add_argument('--no_mul_mat_q', action='store_true', help='Disable the mulmat kernels.')
    group.add_argument('--n_batch', type=int, default=512, help='Maximum number of prompt tokens to batch together when calling llama_eval.')
    group.add_argument('--no-mmap', action='store_true', help='Prevent mmap from being used.')
    group.add_argument('--mlock', action='store_true', help='Force the system to keep the model in RAM.')
    group.add_argument('--n-gpu-layers', type=int, default=0, help='Number of layers to offload to the GPU.')
    group.add_argument('--tensor_split', type=str, default=None, help='Split the model across multiple GPUs. Comma-separated list of proportions. Example: 18,17.')
    group.add_argument('--numa', action='store_true', help='Activate NUMA task allocation for llama.cpp.')
    group.add_argument('--logits_all', action='store_true', help='Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.')
    group.add_argument('--no_offload_kqv', action='store_true', help='Do not offload the  K, Q, V to the GPU. This saves VRAM but reduces the performance.')
    group.add_argument('--cache-capacity', type=str, help='Maximum cache capacity (llama-cpp-python). Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed.')

    # RoPE
    group = parser.add_argument_group('RoPE')
    group.add_argument('--alpha_value', type=float, default=1, help='Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress_pos_emb, not both.')
    group.add_argument('--rope_freq_base', type=int, default=0, help='If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63).')
    group.add_argument('--compress_pos_emb', type=int, default=1, help="Positional embeddings compression factor. Should be set to (context length) / (model\'s original context length). Equal to 1/rope_freq_scale.")

    # Gradio
    group = parser.add_argument_group('Gradio')
    group.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
    group.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    group.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    group.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
    group.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
    group.add_argument('--gradio-auth', type=str, help='Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".', default=None)
    group.add_argument('--gradio-auth-path', type=str, help='Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.', default=None)
    group.add_argument('--ssl-keyfile', type=str, help='The path to the SSL certificate key file.', default=None)
    group.add_argument('--ssl-certfile', type=str, help='The path to the SSL certificate cert file.', default=None)

    return args, deprecated_args

args = parse_arguments()

def do_cmd_flags_warnings():
    
    # Deprecation warnings
    deprecated_args = parse_arguments()
    for k in deprecated_args:
        if getattr(args, k):
            logger.warning(f'The --{k} flag has been deprecated and will be removed soon. Please remove that flag.')

    # Security warnings
    if args.trust_remote_code:
        logger.warning('trust_remote_code is enabled. This is dangerous.')
    if 'COLAB_GPU' not in os.environ and not args.nowebui:
        if args.share:
            logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")
        if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
            logger.warning("\nYou are potentially exposing the web UI to the entire internet without any access password.\nYou can create one with the \"--gradio-auth\" flag like this:\n\n--gradio-auth username:password\n\nMake sure to replace username:password with your own.")
            if args.multi_user:
                logger.warning('\nThe multi-user mode is highly experimental and should not be shared publicly.')


def fix_loader_name(name):
    if not name:
        return name

    name = name.lower()
    if name in ['llamacpp', 'llama.cpp', 'llama-cpp', 'llama cpp']:
        return 'llama.cpp'
    if name in ['llamacpp_hf', 'llama.cpp_hf', 'llama-cpp-hf', 'llamacpp-hf', 'llama.cpp-hf']:
        return 'llamacpp_HF'
    elif name in ['transformers', 'huggingface', 'hf', 'hugging_face', 'hugging face']:
        return 'Transformers'
    elif name in ['ctransformers', 'ctranforemrs', 'ctransformer']:
        return 'ctransformers'

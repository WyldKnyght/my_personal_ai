# user_interface/models_ui.py

import gradio as gr

from common import utils
from user_interface import ui_settings as settings
from user_interface import gradio_ui
from model_handler import model_loaders
from model_handler.default_model_tab_settings import total_mem, default_gpu_mem, total_cpu_mem, default_cpu_mem

def create_models_tab():
    with gr.Tab("Model", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            settings.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: settings.model_name, label='Model', elem_classes='slim-dropdown')
                            gradio_ui.create_refresh_button(settings.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                            settings.gradio['load_model'] = gr.Button("Load", visible=not settings.settings['autoload_model'], elem_classes='refresh-button')
                            settings.gradio['unload_model'] = gr.Button("Unload", elem_classes='refresh-button')
                            settings.gradio['reload_model'] = gr.Button("Reload", elem_classes='refresh-button')
                            settings.gradio['save_model_settings'] = gr.Button("Save settings", elem_classes='refresh-button')

        with gr.Row():
            with gr.Column():
                settings.gradio['loader'] = gr.Dropdown(label="Model loader", choices=model_loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            for i in range(len(total_mem)):
                                settings.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                            settings.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)
                            settings.gradio['transformers_info'] = gr.Markdown('load-in-4bit params:')
                            settings.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=settings.args.compute_dtype)
                            settings.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=settings.args.quant_type)

                            settings.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=128, value=settings.args.n_gpu_layers)
                            settings.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=settings.settings['truncation_length_max'], step=256, label="n_ctx", value=settings.args.n_ctx, info='Context length. Try lowering this if you run out of memory while loading the model.')
                            settings.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=32, value=settings.args.threads)
                            settings.gradio['threads_batch'] = gr.Slider(label="threads_batch", minimum=0, step=1, maximum=32, value=settings.args.threads_batch)
                            settings.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, value=settings.args.n_batch)

                            settings.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=settings.args.wbits if settings.args.wbits > 0 else "None")
                            settings.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=settings.args.groupsize if settings.args.groupsize > 0 else "None")
                            settings.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None"], value=settings.args.model_type or "None")
                            settings.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=settings.args.pre_layer[0] if settings.args.pre_layer is not None else 0)
                            settings.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7')
                            settings.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=0, maximum=settings.settings['truncation_length_max'], step=256, info='Context length. Try lowering this if you run out of memory while loading the model.', value=settings.args.max_seq_len)
                            settings.gradio['alpha_value'] = gr.Slider(label='alpha_value', minimum=1, maximum=8, step=0.05, info='Positional embeddings alpha factor for NTK RoPE scaling. Recommended values (NTKv1): 1.75 for 1.5x context, 2.5 for 2x context. Use either this or compress_pos_emb, not both.', value=settings.args.alpha_value)
                            settings.gradio['rope_freq_base'] = gr.Slider(label='rope_freq_base', minimum=0, maximum=1000000, step=1000, info='If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63)', value=settings.args.rope_freq_base)
                            settings.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb', minimum=1, maximum=8, step=1, info='Positional embeddings compression factor. Should be set to (context length) / (model\'s original context length). Equal to 1/rope_freq_scale.', value=settings.args.compress_pos_emb)
                            settings.gradio['quipsharp_info'] = gr.Markdown('QuIP# has to be installed manually at the moment.')

                        with gr.Column():
                            settings.gradio['tensorcores'] = gr.Checkbox(label="tensorcores", value=settings.args.tensorcores, info='Use llama-cpp-python compiled with tensor cores support. This increases performance on RTX cards. NVIDIA only.')
                            settings.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=settings.args.no_offload_kqv, info='Do not offload the  K, Q, V to the GPU. This saves VRAM but reduces the performance.')
                            settings.gradio['triton'] = gr.Checkbox(label="triton", value=settings.args.triton)
                            settings.gradio['no_inject_fused_attention'] = gr.Checkbox(label="no_inject_fused_attention", value=settings.args.no_inject_fused_attention, info='Disable fused attention. Fused attention improves inference performance but uses more VRAM. Fuses layers for AutoAWQ. Disable if running low on VRAM.')
                            settings.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=settings.args.no_inject_fused_mlp, info='Affects Triton only. Disable fused MLP. Fused MLP improves performance but uses more VRAM. Disable if running low on VRAM.')
                            settings.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=settings.args.no_use_cuda_fp16, info='This can make models faster on some systems.')
                            settings.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=settings.args.desc_act, info='\'desc_act\', \'wbits\', and \'groupsize\' are used for old models without a quantize_config.json.')
                            settings.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=settings.args.no_mul_mat_q, info='Disable the mulmat kernels.')
                            settings.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=settings.args.no_mmap)
                            settings.gradio['mlock'] = gr.Checkbox(label="mlock", value=settings.args.mlock)
                            settings.gradio['numa'] = gr.Checkbox(label="numa", value=settings.args.numa, info='NUMA support can help on some systems with non-uniform memory access.')
                            settings.gradio['cpu'] = gr.Checkbox(label="cpu", value=settings.args.cpu)
                            settings.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=settings.args.load_in_8bit)
                            settings.gradio['bf16'] = gr.Checkbox(label="bf16", value=settings.args.bf16)
                            settings.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=settings.args.auto_devices)
                            settings.gradio['disk'] = gr.Checkbox(label="disk", value=settings.args.disk)
                            settings.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=settings.args.load_in_4bit)
                            settings.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=settings.args.use_double_quant)
                            settings.gradio['tensor_split'] = gr.Textbox(label='tensor_split', info='Split the model across multiple GPUs, comma-separated list of proportions, e.g. 18,17')
                            settings.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=settings.args.trust_remote_code, info='To enable this option, start the web UI with the --trust-remote-code flag. It is necessary for some models.', interactive=settings.args.trust_remote_code)
                            settings.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=settings.args.cfg_cache, info='Create an additional cache for CFG negative prompts.')
                            settings.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=settings.args.logits_all, info='Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.')
                            settings.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=settings.args.use_flash_attention_2, info='Set use_flash_attention_2=True while loading the model.')
                            settings.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama", value=settings.args.disable_exllama, info='Disable ExLlama kernel.')
                            settings.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2", value=settings.args.disable_exllamav2, info='Disable ExLlamav2 kernel.')
                            settings.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=settings.args.no_flash_attn, info='Force flash-attention to not be used.')
                            settings.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=settings.args.cache_8bit, info='Use 8-bit cache to save VRAM.')
                            settings.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=settings.args.no_use_fast, info='Set use_fast=False while loading the tokenizer.')
                            settings.gradio['num_experts_per_token'] = gr.Number(label="Number of experts per token", value=settings.args.num_experts_per_token, info='Only applies to MoE models like Mixtral.')
                            settings.gradio['llamacpp_HF_info'] = gr.Markdown('llamacpp_HF loads llama.cpp as a Transformers model. To use it, you need to download a tokenizer.\n\nOption 1 (recommended): place your .gguf in a subfolder of models/ along with these 4 files: special_tokens_map.json, tokenizer_config.json, tokenizer.json, tokenizer.model.\n\nOption 2: download `oobabooga/llama-tokenizer` under "Download model". That\'s a default Llama tokenizer that will work for some (but not all) models.')

            with gr.Column():
                with gr.Row():
                    settings.gradio['autoload_model'] = gr.Checkbox(value=settings.settings['autoload_model'], label='Autoload the model', info='Whether to load the model as soon as it is selected in the Model dropdown.')
                settings.gradio['custom_model_menu'] = gr.Textbox(label="Download model", info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main. To download a single file, enter its name in the second box.")
                settings.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", show_label=False, max_lines=1)
                with gr.Row():
                    settings.gradio['download_model_button'] = gr.Button("Download", variant='primary')
                    settings.gradio['get_file_list'] = gr.Button("Get file list")

                with gr.Row():
                    settings.gradio['model_status'] = gr.Markdown('No model is loaded' if settings.model_name == 'None' else 'Ready')

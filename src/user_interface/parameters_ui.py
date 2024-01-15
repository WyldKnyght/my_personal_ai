# user_interface/parameters_ui.py

import gradio as gr
from common import utils
from user_interface import ui_settings as settings
from user_interface import gradio_ui

def create_parameters_tab():
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
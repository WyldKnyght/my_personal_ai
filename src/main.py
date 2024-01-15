import os
import signal
import sys
import warnings
from pathlib import Path
from threading import Lock

import gradio as gr
from common import utils
from user_interface import ui_settings
from common.block_requests import OpenMonkeyPatch, RequestBlocker
from common.logging_colors import logger
from model_handler.models import load_model
from model_handler.models_settings import get_model_metadata, update_model_parameters
from user_interface import (
    ai_assistant_ui,
    gradio_ui,
    model_menu_ui,
    parameters_ui,
)

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='The function `run` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.'
)


def signal_handler(sig, frame):
    logger.info("Received Ctrl+C. Shutting down Text generation web UI gracefully.")
    sys.exit(0)


def shutdown_gracefully(sig, frame):
    print("Received Ctrl+C. Shutting down Text generation web UI gracefully.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting AI Assistant web UI")

    models = utils.get_available_models()

    if ui_settings.args.model_menu:
        if len(models) == 0:
            print('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(models)}\n')
            i = int(input()) - 1
            print()

        ui_settings.model_name = models[i]

    if ui_settings.model_name != 'None':
        model_name = Path(ui_settings.model_name).parts[-1]
        ui_settings.model_name = model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings)

        ui_settings.model, ui_settings.tokenizer = load_model(model_name)

    ui_settings.generation_lock = Lock()

    def create_gradio_interface():
        title = 'My Personal AI Assistant'

        with gr.Blocks(analytics_enabled=False, title=title, theme=gradio_ui.theme) as ui_settings.gradio['interface']:
            ui_settings.gradio['temporary_text'] = gr.Textbox(visible=False)

            ai_assistant_ui.create_ai_assistant_tab()
            model_menu_ui.create_models_tab()
            parameters_ui.create_parameters_tab()

            ai_assistant_ui.create_event_handlers()
            model_menu_ui.create_event_handlers()
            parameters_ui.create_event_handlers()

            if ui_settings.settings['dark_theme']:
                ui_settings.gradio['interface'].load(
                    lambda: None,
                    None,
                    None,
                    _js="() => document.getElementsByTagName('body')[0].classList.add('dark')"
                )

            ui_settings.gradio['interface'].queue(concurrency_count=64)


    with OpenMonkeyPatch():
        ui_settings.gradio['interface'].launch(prevent_thread_lock=True)

    create_gradio_interface()
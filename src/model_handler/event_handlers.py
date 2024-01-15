# /src/model_handler/event_handlers.py

from functools import partial
import gradio as gr
from user_interface import gradio_ui
from model_handler.models import load_model, unload_model, update_truncation_length
from model_handler.model_wrapper import load_model_wrapper, unload_model_wrapper, load_custom_model, model_wrapper
from model_download import download_model
from model_handler.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_model_settings,
    update_model_parameters,
)
from common import settings
from common.utils import gradio

def create_event_handlers():
    # Initialize variables
    loader = gradio('loader')
    model_menu = gradio('model_menu')
    interface_state = gradio('interface_state')
    truncation_length = gradio('truncation_length')
    model_status = gradio('model_status')
    custom_model_menu = gradio('custom_model_menu')
    download_specific_file = gradio('download_specific_file')
    autoload_model = gradio('autoload_model')
    input_elements = gradio(settings.input_elements)

    # Define event handlers
    settings.gradio['loader'].change(
        loader.make_loader_params_visible, loader, loader.get_all_params()
    ).then(
        lambda value: gr.update(choices=loader.get_model_types(value)),
        loader,
        settings.model_type
    )
    
    settings.gradio['model_menu'].change(
        gradio_ui.gather_interface_values, input_elements, interface_state
    ).then(
        apply_model_settings_to_state, model_menu, interface_state
    ).then(
        gradio_ui.apply_interface_values, interface_state, gradio_ui.list_interface_input_elements(),
        show_progress=False
    ).then(
        update_model_parameters, interface_state, None
    ).then(
        model_wrapper.load_model_wrapper, model_menu, loader, autoload_model, model_status,
        show_progress=False
    ).success(
        update_truncation_length, truncation_length, interface_state, truncation_length
    ).then(
        lambda x: x, loader, filter_by_loader
    )

    settings.gradio['load_model'].click(
        gradio_ui.gather_interface_values, input_elements, interface_state
    ).then(
        update_model_parameters, interface_state, None
    ).then(
        partial(load_model_wrapper, autoload=True), model_menu, loader, model_status,
        show_progress=False
    ).success(
        update_truncation_length, truncation_length, interface_state, truncation_length
    ).then(
        lambda x: x, loader, filter_by_loader
    )

    settings.gradio['reload_model'].click(
        unload_model, None, None
    ).then(
        gradio_ui.gather_interface_values, input_elements, interface_state
    ).then(
        update_model_parameters, interface_state, None
    ).then(
        partial(load_model_wrapper, autoload=True), model_menu, loader, model_status,
        show_progress=False
    ).success(
        update_truncation_length, truncation_length, interface_state, truncation_length
    ).then(
        lambda x: x, loader, filter_by_loader
    )

    settings.gradio['unload_model'].click(
        unload_model, None, None
    ).then(
        lambda: "Model unloaded", None, model_status
    )

    settings.gradio['save_model_settings'].click(
        gradio_ui.gather_interface_values, input_elements, interface_state
    ).then(
        save_model_settings, model_menu, interface_state, model_status
    )

    settings.gradio['download_model_button'].click(
        download_model_wrapper, download_specific_file, model_status, show_progress=True
    settings.gradio['get_file_list'].click(
        partial(download_model_wrapper, return_links=True), download_specific_file, model_status, show_progress=True
    )

    settings.gradio['autoload_model'].change(
        lambda x: gr.update(visible=not x), autoload_model, load_model
    )
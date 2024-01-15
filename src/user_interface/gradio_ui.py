# user_interface/gradio_ui.py

import gradio as gr
import torch
import yaml
from transformers import is_torch_xpu_available
from user_interface import ui_settings

# Define symbols for buttons
REFRESH_SYMBOL = '🔄'
DELETE_SYMBOL = '🗑️'
SAVE_SYMBOL = '💾'

# Define the default theme for the UI
THEME = gr.themes.Default(
    font=['Noto Sans', 'Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)

def list_model_elements():
    # List all model elements
    elements = [
        'loader',
        'filter_by_loader',
        # ... other elements ...
    ]
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            elements.append(f'gpu_memory_{i}')
    else:
        for i in range(torch.cuda.device_count()):
            elements.append(f'gpu_memory_{i}')

    return elements

def list_interface_input_elements():
    # List all interface input elements
    elements = [
        'max_new_tokens',
        'auto_max_new_tokens',
        # ... other elements ...
    ]

    # Add chat elements
    elements += [
        'textbox',
        'start_with',
        # ... other elements ...
    ]

    # Add model elements
    elements += list_model_elements()

    return elements

def gather_interface_values(*args):
    # Gather interface values into a dictionary
    output = {}
    elements = list_interface_input_elements()
    for i, element in enumerate(elements):
        output[element] = args[i]

    if not ui_settings.args.multi_user:
        ui_settings.persistent_interface_state = output

    return output

def apply_interface_values(state, use_persistent=False):
    # Apply interface values to the state
    if use_persistent:
        state = ui_settings.persistent_interface_state

    elements = list_interface_input_elements()
    return [state.get(k, gr.update()) for k in elements]

def save_settings(state, preset, show_controls, theme_state):
    # Save settings to YAML format
    output = ui_settings.settings.copy()
    exclude = ['name2', 'greeting', 'context', 'turn_template']
    for k in state:
        if k in ui_settings.settings and k not in exclude:
            output[k] = state[k]

    output['preset'] = preset
    output['seed'] = int(output['seed'])
    output['show_controls'] = show_controls
    output['dark_theme'] = theme_state == 'dark'

    return yaml.dump(output, sort_keys=False, width=float("inf"))

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class, interactive=True):
    # Create a refresh button
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
        return gr.update(**(args or {}))

    refresh_button = gr.Button(REFRESH_SYMBOL, elem_classes=elem_class, interactive=interactive)
    refresh_button.click(
        fn=lambda: {k: tuple(v) if type(k) is list else v for k, v in refresh().items()}, inputs=[], outputs=[refresh_component] )

    return refresh_button
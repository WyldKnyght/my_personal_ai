# user_interface/ai_assistant_ui.py

import gradio as gr
from functools import partial
from user_interface import ui_settings
from model_handler.text_generation import stop_everything_event
from model_handler import text_generation
from common.utils import gradio
from user_interface import gradio_ui

inputs = ('Chat input')
clear_arr = ('delete_chat-confirm', 'delete_chat', 'delete_chat-cancel')

def create_ai_assistant_tab():
    ui_settings.gradio['Chat input'] = gr.State()
    ui_settings.gradio['history'] = gr.State({'internal': [], 'visible': []})

    with gr.Tab('Chat', elem_id='chat-tab'):
        with gr.Row():
            with gr.Row(elem_id="chat-input-row"):
                with gr.Column(scale=1, elem_id='gr-hover-container'):
                    gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                with gr.Column(scale=10, elem_id='chat-input-container'):
                    ui_settings.gradio['textbox'] = gr.Textbox(label='', placeholder='Send a message', elem_id='chat-input', elem_classes=['add_scrollbar'])
                    ui_settings.gradio['show_controls'] = gr.Checkbox(value=ui_settings.settings['show_controls'], label='Show controls (Ctrl+S)', elem_id='show-controls')
                    ui_settings.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

                with gr.Column(scale=1, elem_id='generate-stop-container'):
                    with gr.Row():
                        ui_settings.gradio['Stop'] = gr.Button('Stop', elem_id='stop', visible=False)
                        ui_settings.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')

        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                ui_settings.gradio['Regenerate'] = gr.Button('Regenerate (Ctrl + Enter)', elem_id='Regenerate')
                ui_settings.gradio['Continue'] = gr.Button('Continue (Alt + Enter)', elem_id='Continue')

def create_event_handlers():
    ui_settings.gradio['Generate'].click(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements),
        gradio('textbox'),
        gradio('Chat input', 'textbox'),
        show_progress=False
    ).then(
        text_generation.generate_chat_reply_wrapper,
        gradio(inputs),
        gradio('display', 'history'),
        show_progress=False
    ).then(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements)
    )

    ui_settings.gradio['textbox'].submit(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements),
        gradio('interface_state')
    ).then(
        lambda x: (x, ''),
        gradio('textbox'),
        gradio('Chat input', 'textbox'),
        show_progress=False
    ).then(
        text_generation.generate_chat_reply_wrapper,
        gradio(inputs),
        gradio('display', 'history'),
        show_progress=False
    ).then(
        gradio_ui.gather_interface_values,
        gradio (ui_settings.input_elements) )
    ui_settings.gradio['Regenerate'].click(
    gradio_ui.gather_interface_values,
    gradio(ui_settings.input_elements),
    gradio('interface_state')
    ).then(
        partial(text_generation.generate_chat_reply_wrapper, regenerate=True),
        gradio(inputs),
        gradio('display', 'history'),
        show_progress=False
    ).then(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements)
    )

    ui_settings.gradio['Continue'].click(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements),
        gradio('interface_state')
    ).then(
        partial(text_generation.generate_chat_reply_wrapper, _continue=True),
        gradio(inputs),
        gradio('display', 'history'),
        show_progress=False
    ).then(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements)
    )

    ui_settings.gradio['Stop'].click(
        stop_everything_event,
        None,
        None,
        queue=False
    ).then(
        gradio('display')
    )

    ui_settings.gradio['Start new chat'].click(
        gradio_ui.gather_interface_values,
        gradio(ui_settings.input_elements),
        gradio('interface_state')
    ).then(
        text_generation.start_new_chat,
        gradio('interface_state'),
        gradio('history')
    ).then(
        gradio('display')
    )

    ui_settings.gradio['show_controls'].change(
        None,
        gradio('show_controls'),
        None,
        _js=f'(x) => {{{gradio_ui.show_controls_js}; toggle_controls(x)}}'
    )

import re

from user_interface import ui_settings as settings
from model_handler.text_generation import (
    generate_reply,
    get_stopping_strings,
    generate_chat_prompt
)

def chatbot_wrapper(input_text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    chat_history = state['history']

    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = input_text

        chat_history['internal'].append([input_text, ''])
        chat_history['visible'].append([visible_text, ''])

        # *Is typing...*
        if loading_message:
            yield {
                'visible': chat_history['visible'][:-1] + [[chat_history['visible'][-1][0], settings.processing_message]],
                'internal': chat_history['internal']
            }
    else:
        text, visible_text = chat_history['internal'][-1][0], chat_history['visible'][-1][0]
        if regenerate and loading_message:
            yield {
                'visible': chat_history['visible'][:-1] + [[visible_text, settings.processing_message]],
                'internal': chat_history['internal'][:-1] + [[text, '']]
            }
        elif _continue:
            last_reply = [chat_history['internal'][-1][1], chat_history['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': chat_history['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': chat_history['internal']
                }

    if settings.model_name == 'None' or settings.model is None:
        raise ValueError("No model is loaded! Select one in the Model tab.")

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': chat_history if _continue else {k: v[:-1] for k, v in chat_history.items()}
    }
    prompt = generate_chat_prompt(input_text, state, **kwargs)

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

        # Extract the reply
        visible_reply = reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)

        if settings.stop_everything:
            chat_history['visible'][-1][1] = chat_history['visible'][-1][1]
            yield chat_history
            return

        if _continue:
            chat_history['internal'][-1] = [text, last_reply[0] + reply]
            chat_history['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
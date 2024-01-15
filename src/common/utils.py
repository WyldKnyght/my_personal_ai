# src/common/utils.py

import re
from datetime import datetime
from pathlib import Path

from user_interface import ui_settings

# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [ui_settings.gradio[k] for k in keys]

def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

def atoi(text):
    return int(text) if text.isdigit() else text.lower()

# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_available_models():
    model_list = []
    for item in list(Path(f'{ui_settings.args.model_dir}/').glob('*')):
        if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml', '.py')) and 'llama-tokenizer' not in item.name:
            model_list.append(re.sub('.pth$', '', item.name))

    return ['None'] + sorted(model_list, key=natural_keys)
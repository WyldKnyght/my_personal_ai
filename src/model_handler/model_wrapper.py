# /src/model_handlers/model_wrapper.py

from common.logging_colors import logger
from model_handler.models import load_model, unload_model
from model_handler.event_handlers import loaders 
from model_handler.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_model_settings,
    update_model_parameters
)

def load_model_wrapper(selected_model: str, loader: Loader, autoload: bool = False) -> Generator[str, None, None]:
    if not autoload:
        yield f"The settings for '{selected_model}' have been updated.\n\nClick on 'Load' to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading '{selected_model}'..."
            unload_model()
            if selected_model:
                settings.model, settings.tokenizer = load_model(selected_model, loader)

            if settings.model is not None:
                output = f"Successfully loaded '{selected_model}'."

                settings = get_model_metadata(selected_model)
                instruction_template = settings.get('instruction_template')
                if instruction_template:
                    output += f"\n\nIt seems to be an instruction-following model with template '{instruction_template}'. In the chat tab, instruct or chat-instruct modes should be used."

                yield output
            else:
                yield f"Failed to load '{selected_model}'."
        except SomeSpecificException as e:
            logger.error('Failed to load the model: %s', str(e))
            yield str(e)


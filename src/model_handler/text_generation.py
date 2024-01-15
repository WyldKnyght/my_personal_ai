import ast
import copy
import pprint
import random
import re
import time
import traceback
import deepspeed
import numpy as np
import torch
import transformers
from transformers import LogitsProcessorList, is_torch_xpu_available

import user_interface.ui_settings as settings
from model_handler.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
from common.grammar.grammar_utils import initialize_grammar
from common.grammar.logits_process import GrammarConstrainedLogitsProcessor
from common.logging_colors import logger
from model_handler.models import clear_torch_cache, local_rank


def generate_reply(*args, **kwargs):
    settings.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        settings.generation_lock.release()


def select_generation_function():
    if settings.model_name == 'None' or settings.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        return None

    if settings.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel']:
        return generate_reply_custom
    else:
        return generate_reply_HF

def _generate_reply(question, state, stopping_strings=None):
    generate_func = select_generation_function()
    original_question, all_stop_strings, seed, last_update, reply, is_stream, min_update_interval = prepare_all(question, state, stopping_strings)
    generate_reply_and_apply_stopping_strings(generate_func, question, state, stopping_strings, original_question, all_stop_strings, seed, last_update, reply, is_stream, min_update_interval)

def prepare_stopping_strings(stopping_strings, state):
    all_stop_strings = []
    for st in (stopping_strings, state.get('custom_stopping_strings')):
        if isinstance(st, str):
            st = ast.literal_eval(f"[{st}]")

        if isinstance(st, list) and len(st) > 0:
            all_stop_strings += st

    return all_stop_strings

def prepare_state(state):
    seed = int(state['seed'])
    if seed == -1:
        seed = random.randint(1, 2**31)
    return seed

def prepare_all(question, state, stopping_strings):
    original_question = question
    all_stop_strings = prepare_stopping_strings(stopping_strings, state)
    seed = prepare_state(state)
    last_update, reply, is_stream, min_update_interval = prepare_variables(state, all_stop_strings)
    return original_question, all_stop_strings, seed, last_update, reply, is_stream, min_update_interval

def prepare_variables(state, all_stop_strings):
    last_update = -1
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    return last_update, reply, is_stream, min_update_interval

def generate_reply_and_apply_stopping_strings(generate_func, question, state, stopping_strings, original_question, all_stop_strings, seed, last_update, reply, is_stream, min_update_interval):
    for reply in generate_func(question, original_question, seed, state, stopping_strings):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if is_stream:
            cur_time = time.time()
            limit_tokens_per_second(state, cur_time, last_update)
            yield reply

        if stop_found or (state['max_tokens_second'] > 0 and settings.stop_everything):
            break

    yield reply


def limit_tokens_per_second(state, cur_time, last_update, min_update_interval):
    if state['max_tokens_second'] > 0:
        diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
        if diff > 0:
            time.sleep(diff)
        last_update = time.time()
    else:
        if cur_time - last_update > min_update_interval:
            last_update = cur_time
    return last_update

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if settings.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    if settings.model.__class__.__name__ in ['LlamaCppModel', 'CtransformersModel', 'Exllamav2Model']:
        input_ids = settings.tokenizer.encode(str(prompt))
        if settings.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = settings.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)
        if not add_bos_token:
            while len(input_ids[0]) > 0 and input_ids[0][0] == settings.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if settings.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel'] or settings.args.cpu:
        return input_ids
    elif settings.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return input_ids.to(device)
    elif is_torch_xpu_available():
        return input_ids.to("xpu:0")
    else:
        return input_ids.cuda()


def decode(output_ids, skip_special_tokens=True):
    if settings.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    return settings.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_encoded_length(prompt):
    # Encode the prompt
    encoded_prompt = encode(prompt)

    # Get the length of the encoded prompt
    return len(encoded_prompt[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [settings.tokenizer.decode([i]) for i in tokens]

    output = ''
    for row in list(zip(tokens, decoded_tokens)):
        output += f"{str(int(row[0])).ljust(5)}  -  {repr(row[1])}\n"

    return output


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']

def formatted_outputs(reply, model_name):
    # Implement your formatting logic here
    formatted_reply = f"Reply from {model_name}: {reply}"
    return formatted_reply


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    reply = question if not settings.is_seq2seq else ''
    yield formatted_outputs(reply, settings.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not settings.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, settings.model_name)


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)

    return seed


def stop_everything_event():
    settings.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def get_reply_from_output_ids(output_ids, state, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'])

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(settings.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = settings.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply


def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'min_length', 'num_beams', 'length_penalty', 'early_stopping']:
        generate_params[k] = state[k]

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [settings.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not settings.args.no_cache})
    if settings.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((settings.args.cpu, settings.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    inputs_embeds = None
    if inputs_embeds is not None:
       generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [settings.tokenizer.eos_token_id] if settings.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
       grammar = initialize_grammar(state['grammar_string'])
       grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
       processor.append(grammar_processor)

    # No longer applying extensions
    generate_params['logits_processor'] = processor

    if settings.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

    t0 = time.time()
    try:
        if not is_chat and not settings.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = settings.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            starting_from = 0 if settings.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    settings.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if settings.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not settings.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = settings.model.generate(question, state)
            yield reply
        else:
            for reply in settings.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return

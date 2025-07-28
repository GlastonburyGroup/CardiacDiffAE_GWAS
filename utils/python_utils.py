import argparse 
from collections import defaultdict
import re
import numpy as np

class DotDict(dict):
    """
    A custom dictionary that allows accessing values using dot notation.
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

def BoolArgs(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Can be supplied as: ['yes', 'true', 't', 'y', '1'] or ['no', 'false', 'f', 'n', '0']")

def convert_to_int_float_bool(v):
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        try: 
            return BoolArgs(v)
        except argparse.ArgumentTypeError:
            return v

def process_unknown_args(unknown_args):
    unknown_args_dict = {}
    prev_arg = None
    for arg in unknown_args:
        # If the previous argument had no value, then this argument must be the value
        if prev_arg:
            if arg.startswith('-'): #the previous one was a boolean flag
                unknown_args_dict[prev_arg.lstrip('no-')] = not prev_arg.startswith('no-')
                prev_arg = None
            else:
                unknown_args_dict[prev_arg] = convert_to_int_float_bool(arg)
                prev_arg = None
                continue
            
        # Split the argument by the equals sign
        parts = arg.split('=')
        if len(parts) == 2:
            # This arg is a key-value pair
            key, value = parts
            unknown_args_dict[key.lstrip('-')] = convert_to_int_float_bool(value)
        else:
            # This is a flag or a key with no value
            prev_arg = arg.lstrip('-')

    if prev_arg: #If still a previous argument left, then it must be a boolean flag
        unknown_args_dict[prev_arg.lstrip('no-')] = not prev_arg.startswith('no-')
        
    return unknown_args_dict

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def transform_text(s, mode):
    parts = re.split(r'(<\[.*?\]>)', s)
    transformed_parts = []
    
    for part in parts:
        if part.startswith('<[') and part.endswith(']>'):  # If it is wrapped in <[ ]>, case transformation is not applied
            transformed_parts.append(part[2:-2])
        else:
            if mode == 'capitalise_each_word':
                transformed_parts.append(' '.join([word.capitalize() for word in part.split()]))
            elif mode == 'sentence_case':
                transformed_parts.append(part.capitalize())
            elif mode == 'all_lowercase':
                transformed_parts.append(part.lower())
            elif mode == 'all_uppercase':
                transformed_parts.append(part.upper())
            else:
                transformed_parts.append(part)
    
    result = ''
    for i, part in enumerate(transformed_parts):
        if parts[i].startswith('<[') and parts[i].endswith(']>'):
            result += ' ' + part + ' '
        else:
            result += part
    
    return result.strip()

def reduce_numeric_precision(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
    for col in df.select_dtypes(include=['complex128']).columns:
        df[col] = df[col].astype(np.complex64)
    return df
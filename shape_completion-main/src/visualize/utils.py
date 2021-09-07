import os
import pickle
from pathlib import Path

def str_to_bool(value) :
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def userIsSure(opts) -> bool:
        def yes_or_no_ans(question):
            while True:
                reply = str(input(question + ' (y/n): ')).lower().strip()
                if reply[0] == 'y':
                    return True
                if reply[0] == 'n':
                    return False

        print("____________________________________")
        print("parsed params:")
        print(str(opts).split("(")[1].replace(",", "\n").split(")")[0])
        print("____________________________________")
        return yes_or_no_ans("are you sure you want to continue and plot with this params?")

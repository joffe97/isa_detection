import functools
from pathlib import Path
import xxhash
import json
import pickle
from typing import Callable
import logging

from config import Config


def pickle_function_data(data_path: Path, func: Callable[[], object]) -> object:
    if Config.CACHE_DISABLED:
        data = func()
        logging.debug(f"Ignored cache for file: {data_path.absolute()}")
        return data

    if not data_path.exists():
        data = func()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as fid:
            pickle.dump(data, fid)
        logging.debug(f"Dumped data to file: {data_path.absolute()}")
        return data

    with open(data_path, "rb") as fid:
        data = pickle.load(fid)
        logging.debug(f"Loaded data from file: {data_path.absolute()}")
        return data


def pickle_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__qualname__
        args_hash = xxhash.xxh32_hexdigest(
            '|'.join([*args, json.dumps(kwargs, sort_keys=True)]))

        file_path = Config.CACHE_PATH.joinpath(
            "functions", func_name, args_hash)

        return pickle_function_data(file_path, lambda: func(*args, **kwargs))
    return wrapper

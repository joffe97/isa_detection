import functools
import xxhash
import json
import pickle

from config import CACHE_PATH


def pickle_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__qualname__

        args_hash = xxhash.xxh32_hexdigest(
            '|'.join([*args, json.dumps(kwargs, sort_keys=True)]))
        file_path = CACHE_PATH.joinpath(
            "function", func_name, args_hash)

        if file_path.exists():
            with open(file_path, "rb") as fid:
                return pickle.load(fid)

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        result = func(*args, **kwargs)
        with open(file_path, "wb") as fid:
            pickle.dump(result, fid)
        return result
    return wrapper

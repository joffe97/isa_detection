import functools
import json

import xxhash

from config import Config
from domains.caching.cache_file_handler import CacheFileHandler, PickleCacheFileHandler
from .caching import Caching


def cache_func(cache_file_handler: CacheFileHandler = PickleCacheFileHandler()):
    def wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            func_name = func.__qualname__
            args_strs = list(map(str, args))

            if (
                func.__code__.co_argcount != 0
                and func.__code__.co_varnames[0] == "self"
            ):
                func_self = args[0]
                args_strs[0] = json.dumps(func_self.__dict__, sort_keys=True)

            args_hash = xxhash.xxh32_hexdigest(
                "|".join([*args_strs, json.dumps(kwargs, sort_keys=True)])
            )

            file_path = Config.CACHE_PATH.joinpath("functions", func_name, args_hash)

            return Caching(cache_file_handler).load_or_process_func_data(
                file_path, lambda: func(*args, **kwargs)
            )

        return inner_wrapper

    return wrapper

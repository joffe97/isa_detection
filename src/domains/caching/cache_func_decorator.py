import functools
import json

import xxhash

from config import Config
from domains.caching.cache_file_handler import CacheFileHandler, PickleCacheFileHandler
from .caching import Caching


def cache_func(
    cache_file_handler: CacheFileHandler = PickleCacheFileHandler(),
    *,
    use_class_identifier_method: bool = False,
):
    def wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            func_name = func.__qualname__
            first_arg_is_self = func.__code__.co_argcount != 0 and func.__code__.co_varnames[0] == "self"

            if use_class_identifier_method:
                if not first_arg_is_self:
                    raise ValueError(
                        f"cache_func cannot use identifier method if first argument of cached function is not self: {func_name}"
                    )
                func_self = args[0]
                if (identifier_method := getattr(func_self, "identifier", None)) is None or not callable(
                    identifier_method
                ):
                    raise ValueError(
                        f"cache_func cannot use identifier method if the method does not exist: {func_name}"
                    )
                identifier = identifier_method()
            else:
                args_strs = list(map(str, args))
                if first_arg_is_self:
                    func_self = args[0]
                    args_strs[0] = json.dumps(func_self.__dict__, sort_keys=True)
                identifier = xxhash.xxh32_hexdigest(
                    "|".join([*args_strs, json.dumps(kwargs, sort_keys=True)])
                )

            file_path = Config.CACHE_PATH.joinpath("functions", func_name, identifier)

            return Caching(cache_file_handler).load_or_process_func_data(
                file_path, lambda: func(*args, **kwargs)
            )

        return inner_wrapper

    return wrapper

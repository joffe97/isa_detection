import logging
from pathlib import Path
from typing import Any, Callable
import time

from config import Config
from domains.caching.cache_file_handler import CacheFileHandler, PickleCacheFileHandler


class Caching:
    def __init__(
        self, cache_file_handler: CacheFileHandler = PickleCacheFileHandler()
    ) -> None:
        self.cache_file_handler = cache_file_handler

    def load_or_process_func_data(
        self,
        data_path: Path,
        func: Callable[[], Any],
        *,
        cache_disabled=Config.CACHE_DISABLED,
    ) -> Any:
        logging.debug(f"Started cache function for file: {data_path.absolute()}")
        if cache_disabled:
            data = func()
            logging.info(f"Ignored cache for file: {data_path.absolute()}")
            return data

        lock_file_path = Path(".".join([str(data_path), "lock"]))
        if not data_path.exists() or lock_file_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file_path.touch()

            data = func()
            with open(data_path, self.cache_file_handler.write_mode) as fid:
                self.cache_file_handler.dump(data, fid)

            lock_file_path.unlink()
            logging.info(f"Dumped data to file: {data_path.absolute()}")
            return data

        with open(data_path, self.cache_file_handler.read_mode) as fid:
            start_time = time.time()

            data = self.cache_file_handler.load(fid)
            logging.info(f"Loaded data from file: {data_path.absolute()}")
            logging.debug(f"Loaded data from file in {time.time() - start_time} sek")
            return data

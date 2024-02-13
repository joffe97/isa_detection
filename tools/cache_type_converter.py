import path_setup
from domains.feature.feature_entry import FeatureEntry
from domains.caching.cache_file_handler import CacheFileHandler, JsonCacheFileHandler, PickleCacheFileHandler

from pathlib import Path
from typing import Callable
from glob import glob
import argparse


def to_feature_entry(data: dict[str, float]) -> dict[str, FeatureEntry]:
    for key, value in data.items():
        hex_value = key.split("_")[-1]
        dec_value = int(hex_value, base=16)
        feature_entry = FeatureEntry(value, dec_value)
        data[key] = feature_entry
    return data


class CacheTypeConverter:
    def __init__(self, from_cache_file_handler: CacheFileHandler, to_cache_file_handler: CacheFileHandler, data_handler: Callable[[object], object] = None) -> None:
        self.from_cache_file_handler = from_cache_file_handler
        self.to_cache_file_handler = to_cache_file_handler
        self.data_handler = data_handler

    def convert_filepath(self, filepath: Path) -> None:
        if filepath.is_dir():
            return

        filename_tmp = f"{filepath.absolute()}.tmp"
        with open(filepath, self.from_cache_file_handler.read_mode) as f:
            data = self.from_cache_file_handler.load(f)
        filename_tmp_path = filepath.rename(filename_tmp)

        if self.data_handler is not None:
            data = self.data_handler(data)

        with open(filepath, self.to_cache_file_handler.write_mode) as f:
            self.to_cache_file_handler.dump(data, f)

        filename_tmp_path.unlink()
        print(
            f"Successfully changed cache type for file: {filepath.absolute()}")

    def convert_filepaths(self, filepaths: list[Path]) -> None:
        for filepath in filepaths:
            self.convert_filepath(filepath)

    def convert_filenames(self, filenames: list[str]) -> None:
        filepaths = [Path(filename) for filename in filenames]
        self.convert_filepaths(filepaths)

    def convert_filenames_verifier(self, filenames: list[str]) -> None:
        filepaths = [Path(filename) for filename in filenames]
        filepaths_str = '\n'.join([str(filepath.absolute())
                                  for filepath in filepaths])
        continue_input = input(
            f"{filepaths_str}\n\nPlease enter 'y' to parse the files listed above.\n")
        if continue_input.lower() != "y":
            print("Program stopped...")
            return
        self.convert_filepaths(filepaths)


if __name__ == "__main__":
    def cache_file_handler_arg_type(cache_file_handler_str: str) -> CacheFileHandler:
        if cache_file_handler_str == "pickle":
            return PickleCacheFileHandler()
        elif cache_file_handler_str == "json":
            return JsonCacheFileHandler()
        elif cache_file_handler_str.startswith("json") and len(cache_file_handler_str) == 5 and (indent_str := cache_file_handler_str[4]).isnumeric():
            return JsonCacheFileHandler(indent=int(indent_str))
        raise argparse.ArgumentTypeError(
            f"Given CacheFileHandler does not exits: {cache_file_handler_str}! Expected 'json' or 'pickle'.")

    parser = argparse.ArgumentParser()
    parser.add_argument("files")
    parser.add_argument(
        "-f", "--from", type=cache_file_handler_arg_type, required=True)
    parser.add_argument(
        "-t", "--to", type=cache_file_handler_arg_type, required=True)

    args = vars(parser.parse_args())
    files = args["files"]
    from_cache_file_handler = args["from"]
    to_cache_file_handler = args["to"]

    cache_type_converter = CacheTypeConverter(
        from_cache_file_handler, to_cache_file_handler, to_feature_entry).convert_filenames_verifier(glob(files))

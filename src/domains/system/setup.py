import logging
import argparse
from config import Config


class Setup():
    def __init__(self) -> None:
        self.args = self.__setup_parser()

    @staticmethod
    def __setup_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('-log',
                            '--loglevel',
                            default='info',
                            help='Provide logging level. Example --loglevel debug. (default: info)')
        parser.add_argument('--cache',
                            default=True,
                            action=argparse.BooleanOptionalAction,
                            help='Provide option to disable caching.')
        return parser.parse_args()

    def with_logging_config(self) -> "Setup":
        logging.basicConfig(format="%(levelname)s:\t%(message)s",
                            level=self.args.loglevel.upper())
        return self

    def with_cache_config(self) -> "Setup":
        if not self.args.cache:
            Config._disable_cache()
        return self

    def with_all_config(self):
        self.with_cache_config() \
            .with_logging_config()

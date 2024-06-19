import logging
import argparse
from config import Config


class Setup:
    def __init__(self) -> None:
        self.args = self.__setup_parser()

    @staticmethod
    def __setup_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-log",
            "--loglevel",
            default="info",
            help="Provide logging level. Example --loglevel debug. (default: info)",
        )
        parser.add_argument(
            "--cache",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Provide option to disable caching.",
        )
        parser.add_argument(
            "-rs",
            "--result-savers",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Provide option to activate result savers.",
        )
        parser.add_argument(
            "-cpu",
            "--cpu_count",
            default=30,
            type=int,
            help="Maximum CPUs to use. Example -cpu 8. (default: 30)",
        )
        return parser.parse_args()

    def with_logging_config(self) -> "Setup":
        logging.basicConfig(
            format="%(levelname)s:\t%(message)s",
            level=self.args.loglevel.upper(),
        )
        return self

    def with_cache_config(self) -> "Setup":
        if not self.args.cache:
            Config.disable_cache()
        return self

    def with_result_savers_config(self) -> "Setup":
        if self.args.result_savers:
            Config.activate_result_savers()
        return self

    def with_cpu_count_config(self) -> "Setup":
        Config.CPU_CORES = self.args.cpu_count
        return self

    def with_all_config(self) -> "Setup":
        return (
            self.with_cache_config()
            .with_logging_config()
            .with_result_savers_config()
            .with_cpu_count_config()
        )

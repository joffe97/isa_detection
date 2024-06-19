import errno
from typing import Optional, Type
from logging import warning
from multiprocessing import Pool
from config import Config
from domains.caching import cache_func
from domains.feature.features_post_computers import (
    FeaturesPostComputer,
    NoPostComputing,
)
from domains.feature.file_feature_computer_collection import FileFeatureComputer


class FeatureComputerContainer:
    def __init__(
        self,
        file_feature_computer: FileFeatureComputer,
        features_post_computer: FeaturesPostComputer = NoPostComputing(),
    ) -> None:
        self.file_feature_computer = file_feature_computer
        self.features_post_computer = features_post_computer

    def identifier(
        self,
        ignore_features_post_computers: Optional[
            list[Type[FeaturesPostComputer]]
        ] = None,
    ) -> str:
        if ignore_features_post_computers is None:
            ignore_features_post_computers = []

        feature_computers_str = self.file_feature_computer.identifier()
        features_post_computer_str = self.features_post_computer.identifier()

        if any(
            ignore_features_post_computer
            for ignore_features_post_computer in ignore_features_post_computers
            if isinstance(
                self.features_post_computer, ignore_features_post_computer
            )
        ):
            features_post_computer_str = ""

        parts = list(
            filter(None, [feature_computers_str, features_post_computer_str])
        )
        return "+".join(parts)

    def __compute_serial(
        self, binary_files: list[str]
    ) -> list[dict[str, float]]:
        feature_computer_results = [
            self.file_feature_computer.compute(binary_file)
            for binary_file in binary_files
        ]
        return self.features_post_computer.compute(feature_computer_results)

    def __compute_parallel(
        self, binary_files: list[str]
    ) -> list[dict[str, float]]:
        feature_computer_results = []
        retries = 0
        max_retries = 5
        while not feature_computer_results:
            with Pool(Config.CPU_CORES) as p:
                try:
                    feature_computer_results = p.map(
                        self.file_feature_computer.compute, binary_files, 50
                    )
                except BrokenPipeError as e:
                    if retries >= max_retries:
                        raise e
                    retries += 1
                    warning(
                        f"{e}\nRetrying file feature computing ({retries}/{max_retries})..."
                    )

        return self.features_post_computer.compute(feature_computer_results)

    @cache_func(use_class_identifier_method=True)
    def compute(self, binary_files: list[str]) -> list[dict[str, float]]:
        return (
            self.__compute_serial
            if Config.CPU_CORES == 1
            else self.__compute_parallel
        )(binary_files)

from typing import Optional

import pathlib
import logging
import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn import utils as sklearn_utils
from sklearn import exceptions as sklearn_exceptions

from config import Config
from domains.caching.caching import Caching
from domains.model.info.isa_model_result import ISAModelResult


class ISAModel:
    def __init__(
        self,
        architecture_id: int,
        identifier: Optional[str] = None,
        train_test_split: tuple[
            DataFrame, DataFrame, Series, Series
        ] = None,  # type: ignore  # TODO: Make a builder for ISAModel (or something else), to get rid of optional datatypes
        classifier: BaseEstimator = None,  #  type: ignore # TODO: Make a builder for ISAModel (or something else), to get rid of optional datatypes
        architecture_text="Unknown",
    ) -> None:
        self.architecture_id = architecture_id
        self.identifier = identifier
        self.train_test_split = train_test_split
        self.classifier = classifier
        self.__prediction = None
        self.architecture_text = architecture_text

    def with_train_test_split_on_indexes(
        self, data: DataFrame, targets: Series, test_indexes: set[int]
    ) -> "ISAModel":
        test_indexes_list = list(test_indexes)
        train_indexes_list = list(set(range(len(data))).difference(test_indexes))

        x_train = data.iloc[train_indexes_list]
        x_test = data.iloc[test_indexes_list]
        y_train = targets[train_indexes_list]
        y_test = targets[test_indexes_list]

        self.train_test_split = (x_train, x_test, y_train, y_test)
        return self

    @property
    def x_train(self) -> DataFrame:
        return self.train_test_split[0]

    @property
    def x_test(self) -> DataFrame:
        return self.train_test_split[1]

    @property
    def y_train(self) -> Series:
        return self.train_test_split[2]

    @property
    def y_test(self) -> Series:
        return self.train_test_split[3]

    def __get_cache_file_path(self, data_dir: str) -> pathlib.Path:
        if self.identifier is None:
            raise ValueError("self.identifier cannot be None")
        return Config.CACHE_PATH.joinpath(data_dir, self.identifier)

    def __classifier_file_path(self) -> pathlib.Path:
        return self.__get_cache_file_path("classifiers")

    def __precision_file_path(self) -> pathlib.Path:
        return self.__get_cache_file_path("precisions")

    def get_labels(self) -> set[str]:
        labels = set(self.x_train.columns)
        labels.add(str(self.y_train.name))
        return labels

    def train(self) -> None:
        try:
            sklearn_utils.validation.check_is_fitted(self.classifier)
            return
        except sklearn_exceptions.NotFittedError:
            pass

        def fit_and_return_classifier():
            self.classifier.fit(self.x_train, self.y_train)  # type: ignore
            return self.classifier

        self.classifier = Caching().load_or_process_func_data(
            self.__classifier_file_path(), fit_and_return_classifier
        )

    def prediction(self) -> np.ndarray:
        if self.__prediction is None:
            self.train()
            self.__prediction = self.classifier.predict(self.x_test)  # type: ignore
        return self.__prediction

    def precision(self) -> float:
        def calculate_precision():
            correctness = [
                Y == prediction for Y, prediction in zip(self.y_test.to_list(), self.prediction().tolist())
            ]
            return correctness.count(True) / len(correctness)

        precision = Caching().load_or_process_func_data(self.__precision_file_path(), calculate_precision)

        logging.info(f"Found precision for ISAModel with identifier: {self.identifier}")

        return precision

    def find_result(self) -> ISAModelResult:
        precision = self.precision()
        return ISAModelResult(self.architecture_id, precision)

from typing import Optional

import pathlib
import logging
import numpy as np
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin
from sklearn import utils as sklearn_utils
from sklearn import exceptions as sklearn_exceptions

from config import Config
from domains.caching.caching import Caching


class ISAModel:
    def __init__(
        self,
        architecture_id: int,
        identifier: Optional[str] = None,
        train_test_split: tuple[
            DataFrame, DataFrame, Series, Series
        ] = None,  # type: ignore  # TODO: Make a builder for ISAModel (or something else), to get rid of optional datatypes
        classifier: ClassifierMixin = None,  #  type: ignore # TODO: Make a builder for ISAModel (or something else), to get rid of optional datatypes
        architecture_text="Unknown",
    ) -> None:
        self.architecture_id = architecture_id
        self.identifier = identifier
        self.train_test_split = train_test_split
        self.classifier = classifier
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
        # for i in self.train_test_split[0].index:
        #     if self.train_test_split[0].iloc[i].isnull().any():
        #         print(self.train_test_split[0].iloc[i])
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

    def __prediction_file_path(self) -> pathlib.Path:
        return self.__get_cache_file_path("prediction")

    def get_labels(self) -> list[str]:
        labels = list(self.x_train.columns)
        return labels

    def get_possible_values(self) -> set[str]:
        possible_values = set()
        try:
            possible_values = set(self.classifier.classes_)  # type: ignore
        except:
            pass
        return possible_values.union([*self.y_test.values.tolist(), *self.y_train.values.tolist()])

    def train(self) -> None:
        self.get_fitted_classifer()

    def get_fitted_classifer(self) -> ClassifierMixin:
        try:
            sklearn_utils.validation.check_is_fitted(self.classifier)  # type: ignore
            return self.classifier
        except sklearn_exceptions.NotFittedError:
            pass

        def fit_and_return_classifier():
            self.classifier.fit(self.x_train, self.y_train)  # type: ignore
            return self.classifier

        self.classifier = Caching().load_or_process_func_data(
            self.__classifier_file_path(), fit_and_return_classifier
        )
        return self.classifier

    def prediction(self) -> np.ndarray:
        def predict():
            self.train()
            return self.classifier.predict(self.x_test)  # type: ignore

        prediction = Caching().load_or_process_func_data(self.__prediction_file_path(), predict)
        # logging.info(f"Found prediction for ISAModel with identifier: {self.identifier}")
        return prediction

    def prediction_probabilities(self) -> np.ndarray:
        return self.get_fitted_classifer().predict_proba(self.x_test)  # type: ignore

    def precision(self) -> float:
        def calculate_precision():
            predictions = self.prediction().tolist()
            correctness = [Y == prediction for Y, prediction in zip(self.y_test.to_list(), predictions)]
            return correctness.count(True) / len(correctness)

        precision = calculate_precision()
        return precision

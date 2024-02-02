from typing import Optional
from pandas import DataFrame, Series
import pathlib
import sklearn
import numpy as np
from sklearn.base import ClassifierMixin
import logging

from config import Config
from domains.model.info.isa_model_result import ISAModelResult
from helpers import pickle_function_data


class ISAModel:
    def __init__(self, architecture_id: int, identifier: Optional[str] = None, train_test_split: tuple[DataFrame, Series, DataFrame, Series] = None, classifier: ClassifierMixin = None, architecture_text="Unknown") -> None:
        self.architecture_id = architecture_id
        self.identifier = identifier
        self.train_test_split = train_test_split
        self.classifier = classifier
        self.__prediction = None
        self.architecture_text = architecture_text

    def with_train_test_split_on_indexes(self, data: DataFrame, targets: Series, test_indexes: set[int]) -> "ISAModel":
        test_indexes_list = list(test_indexes)
        train_indexes_list = list(
            set(range(len(data))).difference(test_indexes))

        X_train = data.iloc[train_indexes_list]
        Y_train = targets[train_indexes_list]
        X_test = data.iloc[test_indexes_list]
        Y_test = targets[test_indexes_list]

        self.train_test_split = (X_train, X_test, Y_train, Y_test)
        return self

    @property
    def X_train(self) -> DataFrame:
        return self.train_test_split[0]

    @property
    def X_test(self) -> DataFrame:
        return self.train_test_split[1]

    @property
    def Y_train(self) -> Series:
        return self.train_test_split[2]

    @property
    def Y_test(self) -> Series:
        return self.train_test_split[3]

    def __get_cache_file_path(self, data_dir: str) -> Optional[pathlib.Path]:
        if self.identifier is None:
            return None
        return Config.CACHE_PATH.joinpath(data_dir, self.identifier)

    def __classifier_file_path(self) -> Optional[pathlib.Path]:
        return self.__get_cache_file_path("classifiers")

    def __precision_file_path(self) -> Optional[pathlib.Path]:
        return self.__get_cache_file_path("precisions")

    def train(self) -> None:
        try:
            sklearn.utils.validation.check_is_fitted(self.classifier)
            return
        except sklearn.exceptions.NotFittedError:
            pass

        def fit_and_return_classifier():
            self.classifier.fit(self.X_train, self.Y_train)
            return self.classifier

        self.classifier = pickle_function_data(self.__classifier_file_path(),
                                               fit_and_return_classifier)

    def prediction(self) -> np.ndarray:
        if self.__prediction is None:
            self.train()
            self.__prediction = self.classifier.predict(self.X_test)
        return self.__prediction

    def precision(self) -> float:
        def calculate_precision():
            correctness = [Y == prediction for Y, prediction in zip(
                self.Y_test.to_list(), self.prediction().tolist())]
            return correctness.count(True) / len(correctness)

        precision = pickle_function_data(
            self.__precision_file_path(), calculate_precision)

        logging.info(
            f"Found precision for ISAModel with identifier: {self.identifier}")

        return precision

    def find_result(self) -> ISAModelResult:
        precision = self.precision()
        return ISAModelResult(self.architecture_id, precision)

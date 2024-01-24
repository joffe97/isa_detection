from typing import Optional
from pandas import DataFrame, Series
import pathlib
import sklearn
import pickle
import numpy as np

from config import CACHE_PATH


class ISAModel:
    def __init__(self, architecture_id: int, identifier: Optional[str] = None, train_test_split: tuple[DataFrame, Series, DataFrame, Series] = None, classifier: object = None, architecture_text="Unknown") -> None:
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
        return CACHE_PATH.joinpath(data_dir, self.identifier)

    def __classifier_file_path(self) -> Optional[pathlib.Path]:
        return self.__get_cache_file_path("classifiers")

    def __precision_file_path(self) -> Optional[pathlib.Path]:
        return self.__get_cache_file_path("precision")

    def train(self) -> None:
        try:
            sklearn.utils.validation.check_is_fitted(self.classifier)
            return
        except sklearn.exceptions.NotFittedError:
            pass

        model_file_path = self.__classifier_file_path()
        if model_file_path is not None and model_file_path.exists():
            with open(model_file_path, "rb") as fid:
                self.classifier = pickle.load(fid)
                return

        self.classifier.fit(self.X_train, self.Y_train)

        if model_file_path is not None:
            if not model_file_path.parent.exists():
                model_file_path.parent.mkdir()
            with open(model_file_path, "wb") as fid:
                pickle.dump(self.classifier, fid)

    def prediction(self) -> np.ndarray:
        if self.__prediction is None:
            self.train()
            self.__prediction = self.classifier.predict(self.X_test)
        return self.__prediction

    def precision(self) -> float:
        precision_file_path = self.__precision_file_path()
        if precision_file_path is not None and precision_file_path.exists():
            with open(precision_file_path, "rb") as fid:
                return pickle.load(fid)

        correctness = [Y == prediction for Y, prediction in zip(
            self.Y_test.to_list(), self.prediction().tolist())]
        precision = correctness.count(True) / len(correctness)

        if precision_file_path is not None:
            if not precision_file_path.parent.exists():
                precision_file_path.parent.mkdir()
            with open(precision_file_path, "wb") as fid:
                pickle.dump(precision, fid)

        return precision

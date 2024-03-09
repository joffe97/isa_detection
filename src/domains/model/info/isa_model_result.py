import numpy as np
from pandas import Series
from sklearn.base import ClassifierMixin
from domains.model.isa_model import ISAModel


class ISAModelResult:
    def __init__(
        self,
        isa_model: ISAModel,
        prediction: np.ndarray,
    ) -> None:
        self.isa_model = isa_model
        self.prediction = prediction

    @property
    def identifier(self) -> str:
        return self.isa_model.identifier  # type: ignore

    @property
    def architecture_id(self) -> int:
        return self.isa_model.architecture_id

    @property
    def architecture_text(self) -> str:
        return self.isa_model.architecture_text

    @property
    def fitted_classifier(self) -> ClassifierMixin:
        return self.isa_model.get_fitted_classifer()

    @property
    def correct_outputs(self) -> Series:
        return self.isa_model.y_test

    def print(self):
        print(f"{self.architecture_id}\t{self.prediction}")

    def precision(self) -> float:
        def calculate_precision():
            correctness = [
                Y == prediction
                for Y, prediction in zip(self.correct_outputs.to_list(), self.prediction.tolist())
            ]
            return correctness.count(True) / len(correctness)

        precision = calculate_precision()
        return precision

    def get_outputs_set(self) -> set[object]:
        return set(self.correct_outputs)

import numpy as np
from pandas import Series
from sklearn.base import ClassifierMixin
from domains.model.isa_model import ISAModel


class ISAModelResult:
    def __init__(
        self,
        prediction: np.ndarray,
        identifier,
        architecture_id,
        architecture_text,
        # fitted_classifier,
        correct_outputs,
        possible_values,
    ) -> None:
        self.prediction = prediction
        self.identifier = identifier
        self.architecture_id = architecture_id
        self.architecture_text = architecture_text
        # self.fitted_classifier = fitted_classifier
        self.correct_outputs = correct_outputs
        self.possible_values = possible_values

    @classmethod
    def from_isa_model(cls, isa_model: ISAModel) -> "ISAModelResult":
        return cls(
            isa_model.prediction(),
            isa_model.identifier,
            isa_model.architecture_id,
            isa_model.architecture_text,
            # isa_model.get_fitted_classifer(),
            isa_model.y_test,
            isa_model.get_possible_values(),
        )

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

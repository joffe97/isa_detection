import statistics
import xxhash
import sklearn

from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.feature.isa_binary_features import ISABinaryFeatures
from .isa_model import ISAModel


class ISAModelCollection:
    def __init__(
        self,
        classifier: sklearn.base.BaseEstimator,
        isa_models: dict[int, ISAModel] = None,  # type: ignore  # TODO: Fix typing
        identifier: str = None,  # type: ignore  # TODO: Fix typing
    ) -> None:
        self.classifier = classifier
        self.isa_models = isa_models
        self.identifier = identifier

    def __get_classifier_identifier(self) -> str:
        classifier_name = self.classifier.__class__.__name__
        classifier_dict_hash = xxhash.xxh32(str(self.classifier.__dict__)).hexdigest()
        return f"{classifier_name}_{classifier_dict_hash}"

    def with_isa_binary_features(
        self, isa_binary_features: ISABinaryFeatures, clone_classifier=True
    ) -> "ISAModelCollection":
        identifier = f"{isa_binary_features.identifier}/{self.__get_classifier_identifier()}"

        data = isa_binary_features.data
        targets = isa_binary_features.target
        architecture_ids = isa_binary_features.architecture_ids
        architecture_texts = isa_binary_features.architecture_texts
        architecture_id_text_targets = set(zip(architecture_ids, architecture_texts))

        isa_models = dict()
        for architecture_id, architecture_text in architecture_id_text_targets:
            test_indexes = [
                i
                for i, architecture_id_target in enumerate(architecture_ids)
                if architecture_id == architecture_id_target
            ]
            classifier = sklearn.base.clone(self.classifier) if clone_classifier else self.classifier
            isa_models[architecture_id] = ISAModel(
                architecture_id,
                f"{identifier}/{architecture_text}",
                classifier=classifier,
                architecture_text=architecture_texts.iloc[test_indexes[0]],
            ).with_train_test_split_on_indexes(data, targets, set(test_indexes))

        self.isa_models = isa_models
        self.identifier = identifier
        return self

    def train_all(self):
        for isa_model in self.isa_models.values():
            isa_model.train()

    def precisions(self) -> list[float]:
        return [isa_model.precision() for isa_model in self.isa_models.values()]

    def mean_precision(self) -> float:
        return statistics.fmean(self.precisions())

    def print_endianness_precisions(self):
        precisions: dict[int, float] = dict()
        precisions_little = []
        precisions_big = []

        for isa_model in self.isa_models.values():
            cur_precision = isa_model.precision()
            endianness = isa_model.y_test.iloc[0]

            print(
                f"Architecture {isa_model.architecture_id}, \t{endianness}, \t{isa_model.architecture_text},\tprecision: {cur_precision:.4f}"
            )
            precisions[isa_model.architecture_id] = cur_precision

            endianness = isa_model.y_test.iloc[0]
            if endianness == "Little":
                precisions_little.append(cur_precision)
            elif endianness == "Big":
                precisions_big.append(cur_precision)
            else:
                raise ValueError(f"value is not a legal target value: {endianness}")

        print()
        print(
            f"Mean Precision Little:\t{-1 if len(precisions_little) == 0 else statistics.fmean(precisions_little):.4f}"
        )
        print(
            f"Mean Precision Big:\t{-1 if len(precisions_big) == 0 else statistics.fmean(precisions_big):.4f}"
        )
        print()
        print(f"Mean Precision:\t\t{statistics.fmean(precisions.values()):.4f}")

    def find_results(self) -> ISAModelResultCollection:
        return ISAModelResultCollection([isa_model.find_result() for isa_model in self.isa_models.values()])

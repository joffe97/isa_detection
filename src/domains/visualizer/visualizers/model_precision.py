from pandas import DataFrame
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from . import Visualizer


class ModelPrecision(Visualizer):
    @staticmethod
    def get_str(isa_model_info_collection: ISAModelInfoCollection) -> str:
        precision_classifier_dict = dict()
        features = []
        for isa_model_info in isa_model_info_collection.collection:
            features_str = " + ".join(
                isa_model_info.configuration.feature_computer_container_collection.identifiers()
            )
            classifier_str = str(isa_model_info.configuration.classifier)
            files_per_architecture = isa_model_info.configuration.files_per_architecture
            target_label = isa_model_info.configuration.target_label
            row_str = f"{classifier_str}, (FPA={files_per_architecture}, target={target_label})"
            precision = isa_model_info.results.mean_precision()
            precision_classifier_dict.setdefault(row_str, [])
            precision_classifier_dict[row_str].append(precision)
            if features_str not in features:
                features.append(features_str)

        data_frame = DataFrame.from_dict(precision_classifier_dict, columns=features, orient="index")
        title = "Precisions:"
        return "\n".join(map(str, [title, data_frame]))

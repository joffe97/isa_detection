from pandas import DataFrame
import pandas
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from . import Visualizer


class ModelPrecision(Visualizer):
    @staticmethod
    def get_str(isa_model_info_collection: ISAModelInfoCollection) -> str:
        precision_classifier_dict_for_target = dict()
        features_for_target = dict()
        for isa_model_info in isa_model_info_collection.collection:
            features_str = " + ".join(
                isa_model_info.configuration.feature_computer_container_collection.identifiers()
            )
            classifier_str = str(isa_model_info.configuration.classifier)
            files_per_architecture = isa_model_info.configuration.files_per_architecture
            target_label = isa_model_info.configuration.target_label
            row_str = f"{classifier_str}, (FPA={files_per_architecture}, target={target_label})"
            precision = isa_model_info.results.mean_precision()

            precision_classifier_dict_for_target.setdefault(target_label, dict())
            precision_classifier_dict_for_target[target_label].setdefault(row_str, [])
            precision_classifier_dict_for_target[target_label][row_str].append(precision)

            features_for_target.setdefault(target_label, [])
            features = features_for_target[target_label]
            if features_str not in features:
                features.append(features_str)

        target_data_frame_mapping = dict(
            (target, DataFrame.from_dict(precision_classifier_dict, columns=features, orient="index"))
            for target, precision_classifier_dict in precision_classifier_dict_for_target.items()
        )

        display_max_size_pats = ["display.max_rows", "display.max_columns"]
        for shape_index, pat in enumerate(display_max_size_pats):
            pandas.set_option(
                pat,
                max(dataframe.shape[shape_index] for dataframe in target_data_frame_mapping.values()) + 1,
            )
        pandas_display_options = {"display.expand_frame_repr": False}
        for pat, val in pandas_display_options.items():
            pandas.set_option(pat, val)

        target_data_frame_strs = [
            f"- {target}\n{str(data_frame)}" for target, data_frame in target_data_frame_mapping.items()
        ]

        for pat in [*pandas_display_options.keys(), *display_max_size_pats]:
            pandas.reset_option(pat)

        # data_frame = DataFrame.from_dict(
        #     precision_classifier_dict_for_target, columns=features, orient="index"
        # )
        title = "Precisions:"
        return "\n".join([title, "\n\n".join(target_data_frame_strs)])

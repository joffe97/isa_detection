from typing import Callable, Optional
import matplotlib.pyplot as plt

from domains.feature.features_post_computers.keep_specified import KeepSpecified
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.visualizer.result_savers.result_saver import ResultSaver


class ClassifierHyperparamsComparison(ResultSaver):
    def __init__(
        self,
        title: str = "Hyperparameters comparison",
        attribute_to_compare: str = "C",
        configuration_value_to_compare_func: Optional[
            Callable[[ISAModelConfiguration], str]
        ] = None,
        feature_identifier_override: Optional[str] = None,
        y_label_override: str = "Accuracy",
    ) -> None:
        super().__init__()
        self.title = title
        self.attribute_to_compare = attribute_to_compare
        self.configuration_value_to_compare_func = (
            configuration_value_to_compare_func
        )
        self.feature_identifier_override = feature_identifier_override
        self.y_label_override = y_label_override

    def save_in_directory(
        self, isa_model_info_collection: ISAModelInfoCollection
    ) -> None:
        plot_identifier_feature_mapping = dict()
        for isa_model_info in isa_model_info_collection.collection:
            configuration = isa_model_info.configuration
            results = isa_model_info.results

            # target_label = configuration.target_label.value
            plot_identifier = "/".join(
                [
                    configuration.target_label.value,
                    str(configuration.files_per_architecture),
                ]
            )
            classifier_str = str(configuration.classifier)
            classifier_str_split = classifier_str.split("(")
            classifier_str_type = classifier_str_split[0]

            classifier_arg_strs = [
                (arg_value := configuration.classifier.__dict__.get(arg))
                and f"{arg}={arg_value}"
                for arg in ["kernel", "n_neighbors"]
            ]
            readable_classifier_str = ", ".join(
                map(
                    str,
                    filter(None, [classifier_str_type, *classifier_arg_strs]),
                )
            )

            if self.configuration_value_to_compare_func is not None:
                c = self.configuration_value_to_compare_func(configuration)
            else:
                c = configuration.classifier.__dict__.get(
                    self.attribute_to_compare
                )

            feature_identifier = (
                configuration.feature_computer_container_collection.identifier(
                    ignore_features_post_computers=[KeepSpecified]
                )
            ).split("_")[0]

            if feature_identifier.startswith("Fourier"):
                feature_identifier = "Fourier"

            if self.feature_identifier_override is not None:
                feature_identifier = self.feature_identifier_override

            mean_precision = results.mean_precision()

            features_classifier_mapping = (
                plot_identifier_feature_mapping.setdefault(
                    plot_identifier, dict()
                )
            )
            classifier_precision_mapping = (
                features_classifier_mapping.setdefault(
                    feature_identifier, dict()
                )
            )
            c_precision_mapping = classifier_precision_mapping.setdefault(
                readable_classifier_str, dict()
            )
            c_precision_mapping[c] = mean_precision

        # bar_width = 0.2

        for plot_identifier, feature_mapping in sorted(
            plot_identifier_feature_mapping.items()
        ):
            for feature_str, classifier_mapping in sorted(
                feature_mapping.items()
            ):
                _, ax = plt.subplots(figsize=(9, 6))
                for classifier_str, c_precision_mapping in sorted(
                    classifier_mapping.items()
                ):
                    # index = np.arange(len(classifier_mapping))
                    c_list, precisions = list(
                        zip(*sorted(c_precision_mapping.items()))
                    )
                    # if feature_i == 0:
                    #     xticklabels = list(classifier_mapping_sorted.keys())
                    # precisions = list(zip(*classifier_mapping_sorted.values()))
                    # bars = ax.bar(index + feature_i * bar_width, precisions, bar_width, label=feature)
                    ax.plot(c_list, precisions, label=classifier_str)
                    # for rect in bars:
                    #     rect: Rectangle
                    #     height = rect.get_height()
                    #     ax.text(
                    #         rect.get_x() + rect.get_width() / 2.0 + 0.02,
                    #         height + 0.008,
                    #         f"{height:.3f}",
                    #         ha="center",
                    #         va="bottom",
                    #         rotation=90,
                    #         fontsize=7,
                    #     )

                ax.set_xscale("log")
                ax.set_xlabel(self.attribute_to_compare)
                ax.set_ylabel(self.y_label_override)
                ax.set_title(self.title)
                # ax.set_xticks(index + bar_width / 2)
                # ax.set_xticklabels(xticklabels, rotation=90)
                ax.legend(loc="lower left")
                ax.set_ylim(top=1.0)

                fig_path = self._filepath_for_identifier(
                    ".png", f"{plot_identifier}_{feature_str}"
                )
                fig_path.parent.mkdir(parents=True, exist_ok=True)

                plt.savefig(fig_path, bbox_inches="tight")
                plt.close()

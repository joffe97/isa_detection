import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from domains.feature.features_post_computers.keep_specified import KeepSpecified
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.visualizer.result_savers.result_saver import ResultSaver


class PrecisionsBarChart(ResultSaver):
    def __init__(self, title: str = "Model accuracies") -> None:
        super().__init__()
        self.title = title

    def save_in_directory(self, isa_model_info_collection: ISAModelInfoCollection) -> None:
        plot_identifier_feature_mapping = dict()
        for isa_model_info in isa_model_info_collection.collection:
            configuration = isa_model_info.configuration
            results = isa_model_info.results

            plot_identifier = "/".join(
                [configuration.target_label.value, str(configuration.files_per_architecture)]
            )
            all_plot_identifier = "/".join([configuration.target_label.value, "all"])

            classifier_str = str(configuration.classifier)
            classifier_str_split = classifier_str.split("(")
            classifier_str_type = classifier_str_split[0]

            classifier_arg_strs = [
                (arg_value := configuration.classifier.__dict__.get(arg)) and f"{arg}={arg_value}"
                for arg in ["kernel", "n_neighbors"]
            ]
            readable_classifier_str = ", ".join(
                map(str, filter(None, [classifier_str_type, *classifier_arg_strs]))
            )

            feature_identifier = configuration.feature_computer_container_collection.identifier(
                ignore_features_post_computers=[KeepSpecified]
            )

            mean_precision = results.mean_precision()

            for identifier in [plot_identifier, all_plot_identifier]:
                features_classifier_mapping = plot_identifier_feature_mapping.setdefault(identifier, dict())
                classifier_precision_mapping = features_classifier_mapping.setdefault(
                    feature_identifier, dict()
                )
                classifier_precision_mapping[readable_classifier_str] = mean_precision

        bar_width = 0.2

        for plot_identifier, feature_mapping in sorted(plot_identifier_feature_mapping.items()):
            _, ax = plt.subplots(figsize=(9, 6))
            xticklabels = []
            for feature_i, (feature, classifier_mapping) in enumerate(sorted(feature_mapping.items())):
                index = np.arange(len(classifier_mapping))
                classifier_mapping_sorted = dict(sorted(classifier_mapping.items()))
                if feature_i == 0:
                    xticklabels = list(classifier_mapping_sorted.keys())
                precisions = classifier_mapping_sorted.values()
                bars = ax.bar(index + feature_i * bar_width, precisions, bar_width, label=feature)
                for rect in bars:
                    rect: Rectangle
                    height = rect.get_height()
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0 + 0.02,
                        height + 0.008,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=7,
                    )

            ax.set_xlabel("Classifier")
            ax.set_ylabel("Accuracy")
            ax.set_title(self.title)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.legend(loc="lower left")
            ax.set_ylim(top=1.1)

            fig_path = self._filepath_for_identifier(".png", plot_identifier)
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()

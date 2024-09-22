import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.visualizer.result_savers.result_saver import ResultSaver


class ConfusionMatrix(ResultSaver):
    @classmethod
    def _create_and_save_confusion_matrix(
        cls,
        y_true: list,
        y_pred: list,
        labels: list[str],
        architecture_text: str,
        configuration_identifier: str,
        configuration_str: str,
    ) -> None:
        identifier = "/".join([configuration_identifier, architecture_text])
        plot_title = "\n".join([configuration_str, architecture_text])

        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels
        )
        ax = plt.axes()
        ax.set_title(plot_title)
        display.plot(ax=ax)

        cur_path = cls._filepath_for_identifier(".eps", identifier)
        cur_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cur_path)
        plt.close()

    def save_in_directory(
        self, isa_model_info_collection: ISAModelInfoCollection
    ) -> None:
        for isa_model_info in isa_model_info_collection.collection:
            configuration = isa_model_info.configuration
            configuration_identifier = configuration.identifier()
            configuration_str = str(configuration)

            all_y_true = []
            all_y_pred = []
            for result in isa_model_info.results.collection:
                y_test = result.correct_outputs.to_list()
                prediction = result.prediction.tolist()
                labels = list(result.possible_values)

                all_y_true.extend(y_test)
                all_y_pred.extend(prediction)

                self._create_and_save_confusion_matrix(
                    y_test,
                    prediction,
                    labels,
                    result.architecture_text,
                    configuration_identifier,
                    configuration_str,
                )

            self._create_and_save_confusion_matrix(
                all_y_true,
                all_y_pred,
                labels,
                "all",
                configuration_identifier,
                configuration_str,
            )

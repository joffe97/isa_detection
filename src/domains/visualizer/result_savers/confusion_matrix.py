import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from domains.model.info.isa_model_info import ISAModelInfo
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.visualizer.result_savers.result_saver import ResultSaver


class ConfusionMatrix(ResultSaver):
    def save_in_directory(self, isa_model_info_collection: ISAModelInfoCollection) -> None:
        for isa_model_info in isa_model_info_collection.collection:
            configuration = isa_model_info.configuration
            configuration_identifier = configuration.identifier()
            configuration_str = str(configuration)
            for result in isa_model_info.results.collection:
                identifier = "/".join([configuration_identifier, result.architecture_text])
                plot_title = "\n".join([configuration_str, result.architecture_text])

                y_test = result.correct_outputs.to_list()
                prediction = result.prediction.tolist()
                labels = list(result.isa_model.get_possible_values())

                display = ConfusionMatrixDisplay.from_predictions(y_test, prediction, labels=labels)
                # display = ConfusionMatrixDisplay(
                #     confusion_matrix=cur_confusion_matrix
                # )  # , display=classifier.classes_)
                ax = plt.axes()
                ax.set_title(plot_title)
                display.plot(ax=ax)

                cur_path = self._filepath_for_identifier(".png", identifier)
                cur_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(cur_path)
                plt.close()

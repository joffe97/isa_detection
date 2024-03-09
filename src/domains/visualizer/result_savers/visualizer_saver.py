from typing import Type
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.visualizer.result_savers.result_saver import ResultSaver
from domains.visualizer.visualizers.visualizer import Visualizer


class VisualizerSaver(ResultSaver):
    def __init__(self, visualizer: Type[Visualizer]) -> None:
        super().__init__()
        self.visualizer = visualizer

    @classmethod
    def create_list_from_visualizers(cls, visualizers: list[Type[Visualizer]]) -> list["VisualizerSaver"]:
        return list(map(cls, visualizers))

    def save_in_directory(self, isa_model_info_collection: ISAModelInfoCollection) -> None:
        visualizer_str = self.visualizer.get_str(isa_model_info_collection)
        cur_path = self._filepath_for_identifier(".txt", self.visualizer.class_name())
        cur_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cur_path, "w") as f:
            f.write(visualizer_str)

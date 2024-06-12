from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.visualizer.visualizers.visualizer import Visualizer


class ConfigurationInfo(Visualizer):
    @staticmethod
    def get_str(isa_model_info_collection: ISAModelInfoCollection) -> str:
        return "\n\n".join(
            str(info.configuration)
            for info in isa_model_info_collection.collection
        )

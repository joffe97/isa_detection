from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from . import Visualizer


class BaseLine(Visualizer):
    @staticmethod
    def get_str(isa_model_info_collection: ISAModelInfoCollection) -> str:
        classifier_baseline_dict = dict(
            (str(isa_model_info.configuration.target_label), isa_model_info.results.base_line())
            for isa_model_info in isa_model_info_collection.collection
        )
        classifier_baseline_strs = [f"\t{key}: {item}" for key, item in classifier_baseline_dict.items()]
        return "\n".join(
            [
                "Base lines:",
                *classifier_baseline_strs,
            ]
        )

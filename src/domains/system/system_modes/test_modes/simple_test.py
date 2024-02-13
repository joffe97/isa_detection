from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.test_modes.test_mode import TestMode


class SimpleTest(TestMode):
    @staticmethod
    def run(
        isa_model_configuration: ISAModelConfiguration,
        isa_model_collection: ISAModelCollection,
    ) -> ISAModelResultCollection:
        return isa_model_collection.find_results()

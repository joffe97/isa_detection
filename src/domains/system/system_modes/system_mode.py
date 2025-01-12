from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.test_modes.test_mode import TestMode
from domains.system.system_modes.train_modes.train_mode import TrainMode


class SystemMode:
    def __init__(
        self, train_mode: TrainMode, test_mode: TestMode, isa_model_configuration: ISAModelConfiguration
    ) -> None:
        self.train_mode = train_mode
        self.test_mode = test_mode
        self.isa_model_configuration = isa_model_configuration

    def run(self) -> ISAModelResultCollection:
        isa_model_collection = self.train_mode.run(self.isa_model_configuration)
        return self.test_mode.run(self.isa_model_configuration, isa_model_collection)

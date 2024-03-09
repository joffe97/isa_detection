from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_configuration import ISAModelConfiguration


class ISAModelInfo:
    def __init__(self, configuration: ISAModelConfiguration, results: ISAModelResultCollection) -> None:
        self.configuration = configuration
        self.results = results

    def print(self):
        print(f"{self.configuration}\nPrecision: {self.results.mean_precision():.4f}\n")

from domains.model.info.isa_model_info import ISAModelInfo


class ISAModelInfoCollection:
    def __init__(self, collection: list[ISAModelInfo]) -> None:
        self.collection = collection

    def print(self):
        for isa_model_info in self.collection:
            isa_model_info.print()
            print()

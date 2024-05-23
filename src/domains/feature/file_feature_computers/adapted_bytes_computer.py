from domains.caching.cache_func_decorator import cache_func
from domains.feature.bytes_computers.bytes_computer import BytesComputer
from domains.feature.feature_entry import FeatureEntry
from . import FileFeatureComputer


class AdaptedBytesComputer(FileFeatureComputer):
    def __init__(self, bytes_computer: BytesComputer) -> None:
        self.bytes_computer = bytes_computer

    def identifier(self) -> str:
        return "_".join(
            filter(None, [self.bytes_computer.identifier(), self.bytes_computer.get_group_name()])
        )

    @cache_func(use_class_identifier_method=True)
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        with open(binary_file, "rb") as f:
            binary_data = f.read()
        computed_data = self.bytes_computer.compute(binary_data)

        return dict(
            (f"{self.bytes_computer.class_name().lower()}_{index}", FeatureEntry(value, index))
            for index, value in enumerate(computed_data)
        )

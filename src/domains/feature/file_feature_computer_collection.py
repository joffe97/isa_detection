from .file_feature_computers import FileFeatureComputer


class FileFeatureComputerCollection:
    def __init__(self, file_feature_computers: list[FileFeatureComputer]) -> None:
        self.file_feature_computers = file_feature_computers

    def get_feature_computer_strs(self) -> list[str]:
        method_names = [
            feature_computer.__name__ for feature_computer in self.file_feature_computers]
        method_names.sort()
        return method_names

    def get_feature_computer_str(self, seperator=",") -> str:
        return seperator.join(self.get_feature_computer_strs())

    def compute(self, binary_file: str, *, additional_labels: dict[str, object] = dict()) -> tuple[dict[str, object], int]:
        features_list = [feature_computer.compute(
            binary_file) for feature_computer in self.file_feature_computers]

        training_features_length = sum(
            [len(features_part) for features_part in features_list])

        features_list.append(additional_labels)

        features = dict()
        for features_part in features_list:
            features.update(features_part)
        return features, training_features_length
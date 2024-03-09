from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from config import Config

from domains.model.info.isa_model_info_collection import ISAModelInfoCollection


class ResultSaver(ABC):
    @classmethod
    def __class_name(cls) -> str:
        return cls.mro()[0].__name__

    @classmethod
    def _filepath_for_identifier(cls, suffix: str, identifier: Optional[str] = None) -> Path:
        return Config.RESULTS_PATH.joinpath(
            *filter(None, [Config.get_readable_start_datetime(), cls.__class_name(), identifier])
        ).with_suffix(suffix)

    @abstractmethod
    def save_in_directory(self, isa_model_info_collection: ISAModelInfoCollection) -> None:
        pass

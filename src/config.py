import pathlib
import time
import datetime
from typing import Optional


class Config:
    DATA_PATH = pathlib.Path("../data/")

    CACHE_PATH = DATA_PATH.joinpath("cache")
    DATASETS_PATH = DATA_PATH.joinpath("datasets")
    RESULTS_PATH = DATA_PATH.joinpath("results")
    RESEARCH_PATH = DATA_PATH.joinpath("research")

    ISA_DETECT_DATASET_PATH = DATASETS_PATH.joinpath("isa-detect-data")
    CPU_REC_DATASET_PATH = DATASETS_PATH.joinpath("cpu_rec_corpus")
    CORPUS_CLASSIFICATION_PATH = DATASETS_PATH.joinpath(
        "corpus_classification_isa.csv"
    )
    CUSTOM_DATASET_PATH = DATASETS_PATH.joinpath("custom")

    CACHE_DISABLED = False
    RUN_RESULT_SAVERS = False

    START_TIME = time.time()

    __runtime_identifier = None

    @classmethod
    def disable_cache(cls):
        cls.CACHE_DISABLED = True

    @classmethod
    def activate_result_savers(cls):
        cls.RUN_RESULT_SAVERS = True

    @classmethod
    def get_start_datetime(cls) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(cls.START_TIME)

    @classmethod
    def get_readable_start_datetime(cls) -> str:
        start_datetime = cls.get_start_datetime()
        year = str(start_datetime.year)
        month = str(start_datetime.month).zfill(2)
        day = str(start_datetime.day).zfill(2)
        hour = str(start_datetime.hour).zfill(2)
        minute = str(start_datetime.minute).zfill(2)
        second = str(start_datetime.second).zfill(2)
        return f"{year}{month}{day}_{hour}{minute}{second}"

    @classmethod
    def set_runtime_identifier(cls, runtime_identifier: Optional[str]):
        cls.__runtime_identifier = runtime_identifier

    @classmethod
    def get_runtime_identifier(cls) -> str:
        return cls.__runtime_identifier or cls.get_readable_start_datetime()

import pathlib


class Config:
    DATA_PATH = pathlib.Path("../data/")

    CACHE_PATH = DATA_PATH.joinpath("cache")
    DATASETS_PATH = DATA_PATH.joinpath("datasets")

    ISA_DETECT_DATASET_PATH = DATASETS_PATH.joinpath("isa-detect-data")
    CPU_REC_DATASET_PATH = DATASETS_PATH.joinpath("cpu_rec_corpus")
    CORPUS_CLASSIFICATION_PATH = DATASETS_PATH.joinpath("corpus_classification_isa.csv")

    CACHE_DISABLED = False

    @classmethod
    def disable_cache(cls):
        cls.CACHE_DISABLED = True

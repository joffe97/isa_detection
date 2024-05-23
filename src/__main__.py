from typing import Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from domains.system.setup import Setup

Setup().with_all_config()
from config import Config

from domains.dataset import IsaDetectCode, IsaDetect, CpuRec
from domains.visualizer.result_savers.visualizer_saver import VisualizerSaver
from domains.feature.feature_computer_container import FeatureComputerContainer
from domains.feature.feature_computer_container_collection import (
    FeatureComputerContainerCollection,
)
from domains.feature.features_post_computers import FeaturesPostComputer, MostCommon

from domains.feature.file_feature_computer_collection import FileFeatureComputer
from domains.label.label_entry import LabelEntry
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes import SystemMode
from domains.system.system_modes.test_modes import SimpleTest, DatasetTest
from domains.system.system_modes.train_modes import DatasetTrain
from domains.system.system import System
from domains.feature.file_feature_computers import (
    TrigramsNonZero,
    Trigrams,
    ByteFrequencyDistribution,
    Bigrams,
    EndiannessSignatures,
    ByteDifference,
    BigramDifference,
    AutoCorrelation,
    AutoCorrelationChunks,
    AutoCorrelations,
    AdaptedBytesComputer,
)
from domains.feature.bytes_computers import AutoCorrelationPeakComputer, AutoCorrelationComputer
from domains.visualizer.visualizers import Visualizer, ModelPrecision, BaseLine, StandardDeviations
from domains.visualizer.result_savers import ResultSaver, ConfusionMatrix


def run_system():
    # system_mode = SystemMode(DatasetTrain(IsaDetectCode), SimpleTest())
    # system_mode = SystemMode(DatasetTrain(IsaDetectCode), DatasetTest(CpuRec()))
    # system_mode = SystemMode(DatasetTrain(IsaDetectCode), DatasetTest(IsaDetect(10)))
    # system_mode = SystemMode(DatasetTrain(IsaDetect), DatasetTest(IsaDetectCode(10)))
    # system_mode = SystemMode(DatasetTrain(IsaDetect), SimpleTest())
    system_mode = SystemMode(DatasetTrain(CpuRec), SimpleTest())
    # system_mode = SystemMode(DatasetTrain(CpuRec), DatasetTest(IsaDetectCode(50)))

    feature_computer_container_params: list[list] = [
        # [AutoCorrelationChunks(chunk_size=1024, chunk_count=1, lag=1)],
        # [BigramDifference()],
        # [AutoCorrelation(8)],
        # [AutoCorrelation(4)],
        # [AutoCorrelation(), ByteFrequencyDistribution()],
        # [AutoCorrelations(list(range(1, 16 + 1)))],
        [AdaptedBytesComputer(AutoCorrelationComputer(16))],
        [AdaptedBytesComputer(AutoCorrelationComputer(32))],
        [AdaptedBytesComputer(AutoCorrelationComputer(128))],
        # [AdaptedBytesComputer(AutoCorrelationPeakComputer(32, n_peaks=8))],
        # [AdaptedBytesComputer(AutoCorrelationPeakComputer(64, n_peaks=8))],
        # [ByteDifference, Bigrams],
        # [ByteDifference()],
        # [(TrigramsNonZero, MostCommon(1000))],
        # [Trigrams()],
        # [ByteFrequencyDistribution()],
        # [Bigrams()],
        # [EndiannessSignatures()],
        # [EndiannessSignatures(), ByteFrequencyDistribution()],
    ]
    feature_computer_container_collections = [
        FeatureComputerContainerCollection(
            [
                (
                    FeatureComputerContainer(*feature)  # pylint: disable=E1133
                    if (isinstance(feature, tuple))
                    else FeatureComputerContainer(feature)
                )
                for feature in feature_list
            ]
        )
        for feature_list in feature_computer_container_params
    ]

    RANDOM_STATE = 42
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        LogisticRegression(max_iter=10_000, n_jobs=-1, random_state=RANDOM_STATE, C=100_000_000),
        SVC(kernel="linear", random_state=RANDOM_STATE, C=1_000_000),
        SVC(kernel="poly", random_state=RANDOM_STATE, C=100_000),
        SVC(kernel="sigmoid", random_state=RANDOM_STATE, C=0.01),
        SVC(kernel="rbf", random_state=RANDOM_STATE, C=10_000),
        GaussianNB(),
        KNeighborsClassifier(1, n_jobs=-1),
        KNeighborsClassifier(3, n_jobs=-1),
        KNeighborsClassifier(5, n_jobs=-1),
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        # MLPClassifier(max_iter=10_000, random_state=random_state),
    ]

    target_labels = [
        # LabelEntry.WORD_SIZE,
        # LabelEntry.ENDIANNESS,
        LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
        LabelEntry.FIXED_INSTRUCTION_SIZE,
        # LabelEntry.WORD_SIZE
    ]

    files_per_architecture_list = [10]

    isa_model_configurations = ISAModelConfiguration.create_every_combination(
        feature_computer_container_collections,
        classifiers,
        files_per_architecture_list,
        target_labels,
    )

    visualizers: list[Type[Visualizer]] = [BaseLine, ModelPrecision]
    result_savers: list[ResultSaver] = [
        *VisualizerSaver.create_list_from_visualizers(visualizers),
        ConfusionMatrix(),
    ]

    System(system_mode, isa_model_configurations).run_and_visualize(visualizers, result_savers)


if __name__ == "__main__":
    run_system()

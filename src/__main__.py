from typing import Optional, Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from domains.system.setup import Setup
from domains.visualizer.result_savers.precisions_bar_chart import PrecisionsBarChart

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
from domains.visualizer.result_savers import ResultSaver, ConfusionMatrix, ClassifierHyperparamsComparison


def run_system():
    system_mode = SystemMode(DatasetTrain(IsaDetectCode), SimpleTest())
    # system_mode = SystemMode(DatasetTrain(IsaDetectCode), DatasetTest(CpuRec()))
    # system_mode = SystemMode(DatasetTrain(IsaDetectCode), DatasetTest(IsaDetect(10)))
    # system_mode = SystemMode(DatasetTrain(IsaDetect), DatasetTest(IsaDetectCode(10)))
    # system_mode = SystemMode(DatasetTrain(IsaDetect), SimpleTest())
    # system_mode = SystemMode(DatasetTrain(CpuRec), SimpleTest())
    # system_mode = SystemMode(DatasetTrain(CpuRec), DatasetTest(IsaDetectCode(10)))
    # system_mode = SystemMode(DatasetTrain(CpuRec), DatasetTest(IsaDetectCode(50)))

    # feature_computer_container_params: list[list] = [
    #     # [AutoCorrelationChunks(chunk_size=1024, chunk_count=1, lag=1)],
    #     # [BigramDifference()],
    #     # [AutoCorrelation(8)],
    #     # [AutoCorrelation(4)],
    #     # [AutoCorrelation(), ByteFrequencyDistribution()],
    #     # [AutoCorrelations(list(range(1, 16 + 1)))],
    #     # [AdaptedBytesComputer(AutoCorrelationComputer(16))],
    #     # [AdaptedBytesComputer(AutoCorrelationComputer(32))],
    #     # [AdaptedBytesComputer(AutoCorrelationComputer(128))],
    #     # [AdaptedBytesComputer(AutoCorrelationPeakComputer(32, n_peaks=8))],
    #     # [AdaptedBytesComputer(AutoCorrelationPeakComputer(64, n_peaks=8))],
    #     # [ByteDifference, Bigrams],
    #     # [ByteDifference()],
    #     # [(TrigramsNonZero, MostCommon(1000))],
    #     # [Trigrams()],
    #     [ByteFrequencyDistribution()],
    #     [Bigrams()],
    #     [EndiannessSignatures()],
    #     [EndiannessSignatures(), ByteFrequencyDistribution()],
    # ]

    def create_feature_computer_container_collections(feature_computer_container_params: list[list]):
        return [
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

    # feature_computer_container_collections = create_feature_computer_container_collections(
    #     feature_computer_container_params
    # )

    RANDOM_STATE = 42
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        GaussianNB(),
        KNeighborsClassifier(1, n_jobs=-1),
        KNeighborsClassifier(3, n_jobs=-1),
        KNeighborsClassifier(5, n_jobs=-1),
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        MLPClassifier(max_iter=10_000, random_state=RANDOM_STATE),
    ]

    def get_hyperparam_classifiers(lr_c, svc_l_c, svc_p_c, svc_s_c, svc_r_c):
        return [
            LogisticRegression(max_iter=10_000, n_jobs=-1, random_state=RANDOM_STATE, C=lr_c),
            SVC(kernel="linear", random_state=RANDOM_STATE, C=svc_l_c),
            # SVC(kernel="poly", random_state=RANDOM_STATE, C=svc_p_c),
            # SVC(kernel="sigmoid", random_state=RANDOM_STATE, C=svc_s_c),
            # SVC(kernel="rbf", random_state=RANDOM_STATE, C=svc_r_c),
        ]

    def get_classifiers(lr_c, svc_l_c, svc_p_c, svc_s_c, svc_r_c):
        return [*classifiers, *get_hyperparam_classifiers(lr_c, svc_l_c, svc_p_c, svc_s_c, svc_r_c)]

    # target_labels = [
    #     # LabelEntry.WORD_SIZE,
    #     # LabelEntry.ENDIANNESS,
    #     LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
    #     # LabelEntry.FIXED_INSTRUCTION_SIZE,
    #     # LabelEntry.WORD_SIZE
    # ]

    # files_per_architecture_list: list[Optional[int]] = [None]
    files_per_architecture_list: list[Optional[int]] = [150]
    # files_per_architecture_list: list[Optional[int]] = [3]

    # isa_model_configurations = ISAModelConfiguration.create_every_combination(
    #     feature_computer_container_collections,
    #     classifiers,
    #     files_per_architecture_list,
    #     target_labels,
    # )
    model_configuration_data = [
        # (
        #     AdaptedBytesComputer(AutoCorrelationComputer(32)),
        #     # ByteDifference(),
        #     get_classifiers(10**1, 10**1, 10**1, 10**0, 10**0),
        #     LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
        # ),
        # (
        #     AdaptedBytesComputer(AutoCorrelationComputer(32)),
        #     # ByteDifference(),
        #     get_classifiers(10**2, 10**1, 10**3, 10**1, 10**2),
        #     LabelEntry.FIXED_INSTRUCTION_SIZE,
        # ),
        # (
        #     EndiannessSignatures(),
        #     get_classifiers(10**10, 10**8, 10**5, 10**7, 10**0),
        #     LabelEntry.ENDIANNESS,
        # ),
        (
            Bigrams(),
            get_classifiers(10**7, 10**4, 10**2, 10**1, 10 ** (-1)),
            LabelEntry.ENDIANNESS,
        ),
    ]

    def create_isa_model_configuration_for_data(data):
        return ISAModelConfiguration.create_every_combination(
            [FeatureComputerContainerCollection([FeatureComputerContainer(data[0])])],
            data[1],
            files_per_architecture_list,
            [data[2]],
        )

    # isa_model_configurations_is_variable_instruction_size = ISAModelConfiguration.create_every_combination(
    #     [
    #         FeatureComputerContainerCollection(
    #             [FeatureComputerContainer(AdaptedBytesComputer(AutoCorrelationComputer(32)))]
    #         )
    #     ],
    #     get_classifiers(10**1, 10**1, 10**1, 10**0, 10**0),
    #     files_per_architecture_list,
    #     [LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE],
    # )

    isa_model_configurations = []
    for data in model_configuration_data:
        isa_model_configurations.extend(create_isa_model_configuration_for_data(data))

    visualizers: list[Type[Visualizer]] = [BaseLine, ModelPrecision]
    result_savers: list[ResultSaver] = [
        *VisualizerSaver.create_list_from_visualizers(visualizers),
        ConfusionMatrix(),
        PrecisionsBarChart("Classifiers with features, accuracy"),
        # ClassifierHyperparamsComparison(),
    ]

    System(system_mode, isa_model_configurations).run_and_visualize(visualizers, result_savers)


if __name__ == "__main__":
    run_system()

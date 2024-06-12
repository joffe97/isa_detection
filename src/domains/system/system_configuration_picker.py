from typing import Callable, Optional, Type, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config
from domains.dataset.cpu_rec import CpuRec
from domains.dataset.isa_detect import IsaDetect
from domains.dataset.isa_detect_code import IsaDetectCode
from domains.feature.feature_computer_container import FeatureComputerContainer
from domains.feature.feature_computer_container_collection import (
    FeatureComputerContainerCollection,
)
from domains.feature.file_feature_computers import (
    ByteDifferencePrimes,
    Bigrams,
    AdaptedBytesComputer,
    EndiannessSignatures,
)
from domains.feature.bytes_computers import (
    AutoCorrelationComputer,
    FourierComputer,
    AutoCorrelationPeakComputer,
)
from domains.label.label_entry import LabelEntry
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system import System
from domains.system.system_modes.system_mode import SystemMode
from domains.system.system_modes.test_modes.dataset_test import DatasetTest
from domains.system.system_modes.test_modes.simple_test import SimpleTest
from domains.system.system_modes.test_modes.test_mode import TestMode
from domains.system.system_modes.train_modes.dataset_train import DatasetTrain
from domains.visualizer.result_savers.classifier_hyperparams_comparison import (
    ClassifierHyperparamsComparison,
)
from domains.visualizer.result_savers.confusion_matrix import ConfusionMatrix
from domains.visualizer.result_savers.precisions_bar_chart import (
    PrecisionsBarChart,
)
from domains.visualizer.result_savers.result_saver import ResultSaver
from domains.visualizer.result_savers.visualizer_saver import VisualizerSaver
from domains.visualizer.visualizers.base_line import BaseLine
from domains.visualizer.visualizers.configuration_info import ConfigurationInfo
from domains.visualizer.visualizers.visualizer import Visualizer


class SystemConfiguration:
    def __init__(
        self,
        system_mode_group_datas: list[tuple[str, list[tuple]]],
        visualizers: list[Type[Visualizer]],
        result_savers: list[ResultSaver],
    ) -> None:
        self.system_mode_group_datas = system_mode_group_datas
        self.visualizers = visualizers
        self.result_savers = result_savers

    @classmethod
    def with_precision_visres(
        cls,
        system_mode_group_datas: list[tuple[str, list[tuple]]],
    ) -> "SystemConfiguration":
        visualizers: list[Type[Visualizer]] = [BaseLine]
        result_savers = [
            *VisualizerSaver.create_list_from_visualizers(
                [*visualizers, ConfigurationInfo]
            ),
            PrecisionsBarChart("Classifiers with features, accuracy"),
            ConfusionMatrix(),
        ]
        return cls(
            system_mode_group_datas,
            visualizers,
            result_savers,
        )

    @classmethod
    def with_paramcomp_visres(
        cls,
        plot_title: str,
        configuration_value_to_compare_func: Callable[
            [ISAModelConfiguration], str
        ],
        feature_identifier_override: str,
        x_label: str,
        system_mode_group_datas: list[tuple[str, list[tuple]]],
    ) -> "SystemConfiguration":
        visualizers: list[Type[Visualizer]] = [BaseLine]
        result_savers = [
            *VisualizerSaver.create_list_from_visualizers(
                [*visualizers, ConfigurationInfo]
            ),
            ClassifierHyperparamsComparison(
                plot_title,
                attribute_to_compare=x_label,
                configuration_value_to_compare_func=configuration_value_to_compare_func,
                feature_identifier_override=feature_identifier_override,
            ),
        ]
        return cls(
            system_mode_group_datas,
            visualizers,
            result_savers,
        )

    @classmethod
    def with_hyperparam_visres(
        cls,
        system_mode_group_datas: list[tuple[str, list[tuple]]],
    ) -> "SystemConfiguration":
        visualizers: list[Type[Visualizer]] = [BaseLine]
        result_savers = [
            *VisualizerSaver.create_list_from_visualizers(
                [*visualizers, ConfigurationInfo]
            ),
            ClassifierHyperparamsComparison(),
        ]
        return cls(
            system_mode_group_datas,
            visualizers,
            result_savers,
        )

    @staticmethod
    def create_system_modes_for_data(
        data: tuple,
    ) -> list[SystemMode]:
        return [
            SystemMode(data[0], data[1], configuration)
            for configuration in ISAModelConfiguration.create_every_combination(
                [
                    FeatureComputerContainerCollection(
                        [FeatureComputerContainer(data[2])]
                    )
                ],
                data[3],
                [data[4]],
                [data[5]],
            )
        ]

    def get_system_mode_lists(self) -> list[tuple[str, list[SystemMode]]]:
        system_mode_lists = []
        for (
            runtime_identifier,
            system_mode_data_group,
        ) in self.system_mode_group_datas:
            system_mode_group = []
            for system_mode_data in system_mode_data_group:
                system_mode_group.extend(
                    self.create_system_modes_for_data(system_mode_data),
                )
            system_mode_lists.append((runtime_identifier, system_mode_group))
        return system_mode_lists

    def run_system(self):
        for (
            runtime_identifier,
            system_mode_group,
        ) in self.get_system_mode_lists():
            Config.set_runtime_identifier(runtime_identifier)
            print(f"Starting: {runtime_identifier}")
            System(system_mode_group).run_and_visualize(
                self.visualizers, self.result_savers
            )


class SystemConfigurationPicker:
    RANDOM_STATE = 42

    @classmethod
    def get_hyperparam_classifiers(
        cls,
        lr_c: Optional[Union[int, float]],
        svc_l_c: Optional[Union[int, float]],
    ):
        hyperparam_classifiers = []
        if lr_c is not None:
            hyperparam_classifiers.append(
                LogisticRegression(
                    max_iter=10_000,
                    n_jobs=-1,
                    random_state=cls.RANDOM_STATE,
                    C=lr_c,
                )
            )
        if svc_l_c is not None:
            hyperparam_classifiers.append(
                SVC(kernel="linear", random_state=cls.RANDOM_STATE, C=svc_l_c)
            )
        return hyperparam_classifiers

    @classmethod
    def get_classifiers(
        cls,
        lr_c: Optional[Union[int, float]],
        svc_l_c: Optional[Union[int, float]],
    ):
        return [
            *cls.get_hyperparam_classifiers(lr_c, svc_l_c),
            RandomForestClassifier(n_jobs=-1, random_state=cls.RANDOM_STATE),
            GaussianNB(),
            KNeighborsClassifier(1, n_jobs=-1),
            KNeighborsClassifier(3, n_jobs=-1),
            KNeighborsClassifier(5, n_jobs=-1),
            DecisionTreeClassifier(random_state=cls.RANDOM_STATE),
            MLPClassifier(max_iter=10_000, random_state=cls.RANDOM_STATE),
        ]

    @classmethod
    def hyperparams_endianness_full_10_20(cls) -> SystemConfiguration:
        return SystemConfiguration.with_hyperparam_visres(
            [
                (
                    "hyperparams_endianness_isadetect_30",
                    [
                        *[
                            (
                                DatasetTrain(IsaDetect),
                                SimpleTest(),
                                Bigrams(),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                30,
                                LabelEntry.ENDIANNESS,
                            )
                            for c in list(range(1, 11))
                        ],
                        *[
                            (
                                DatasetTrain(IsaDetect),
                                SimpleTest(),
                                EndiannessSignatures(),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                30,
                                LabelEntry.ENDIANNESS,
                            )
                            for c in list(range(1, 12))
                        ],
                    ],
                ),
                (
                    "hyperparams_endianness_isadetect_100",
                    [
                        (
                            DatasetTrain(IsaDetect),
                            SimpleTest(),
                            EndiannessSignatures(),
                            cls.get_hyperparam_classifiers(10**c, 10**c),
                            100,
                            LabelEntry.ENDIANNESS,
                        )
                        for c in list(range(1, 12))
                    ],
                ),
            ],
        )

    @classmethod
    def hyperparams_endianness_code(cls) -> SystemConfiguration:
        return SystemConfiguration.with_hyperparam_visres(
            [
                (
                    "hyperparams_endianness_code",
                    [
                        *[
                            (
                                DatasetTrain(IsaDetectCode),
                                SimpleTest(),
                                EndiannessSignatures(),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                30,
                                LabelEntry.ENDIANNESS,
                            )
                            for c in list(range(1, 12))
                        ],
                        *[
                            (
                                DatasetTrain(IsaDetectCode),
                                SimpleTest(),
                                Bigrams(),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                30,
                                LabelEntry.ENDIANNESS,
                            )
                            for c in list(range(1, 11))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def hyperparams_isvar(cls) -> SystemConfiguration:
        return SystemConfiguration.with_hyperparam_visres(
            [
                (
                    "hyperparams_isvar",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                ByteDifferencePrimes(),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-4, 7))
                        ],
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(
                                    AutoCorrelationComputer(128)
                                ),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-1, 7))
                        ],
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(FourierComputer(32)),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-2, 7))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def hyperparams_instsize(cls) -> SystemConfiguration:
        return SystemConfiguration.with_hyperparam_visres(
            [
                (
                    "hyperparams_instsize",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(
                                    AutoCorrelationComputer(128)
                                ),
                                cls.get_hyperparam_classifiers(10**c, None),
                                None,
                                LabelEntry.FIXED_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-7, 8))
                        ],
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(
                                    AutoCorrelationComputer(64)
                                ),
                                cls.get_hyperparam_classifiers(None, 10**c),
                                None,
                                LabelEntry.FIXED_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-7, 8))
                        ],
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(FourierComputer(128)),
                                cls.get_hyperparam_classifiers(10**c, 10**c),
                                None,
                                LabelEntry.FIXED_INSTRUCTION_SIZE,
                            )
                            for c in list(range(-7, 8))
                        ],
                        # *[
                        #     (
                        #         DatasetTrain(CpuRec),
                        #         SimpleTest(),
                        #         AdaptedBytesComputer(
                        #             AutoCorrelationPeakComputer(32, n_peaks=3)
                        #         ),
                        #         cls.get_hyperparam_classifiers(10**c, 10**c),
                        #         None,
                        #         LabelEntry.FIXED_INSTRUCTION_SIZE,
                        #     )
                        #     for c in list(range(-7, 8))
                        # ],
                    ],
                ),
            ],
        )

    @classmethod
    def isvar_autocorr_comparison(cls) -> SystemConfiguration:
        def get_lagmax(configuration: ISAModelConfiguration) -> str:
            return configuration.feature_computer_container_collection.feature_computer_containers[
                0
            ].file_feature_computer.bytes_computer.lag_max  # type: ignore

        return SystemConfiguration.with_paramcomp_visres(
            "Autocorrelation lag range comparison",
            get_lagmax,
            "AutoCorrelation",
            "lag",
            [
                (
                    "isvar_autocorr_comparison",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(
                                    AutoCorrelationComputer(2**c)
                                ),
                                cls.get_hyperparam_classifiers(10**1, 10**1),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(1, 11))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def instsize_autocorr_comparison(cls) -> SystemConfiguration:
        def get_lagmax(configuration: ISAModelConfiguration) -> str:
            return configuration.feature_computer_container_collection.feature_computer_containers[
                0
            ].file_feature_computer.bytes_computer.lag_max  # type: ignore

        return SystemConfiguration.with_paramcomp_visres(
            "Autocorrelation lag range comparison",
            get_lagmax,
            "AutoCorrelation",
            "lag",
            [
                (
                    "instsize_autocorr_comparison",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(
                                    AutoCorrelationComputer(2**c)
                                ),
                                cls.get_hyperparam_classifiers(10**1, 10**1),
                                None,
                                LabelEntry.INSTRUCTION_SIZE,
                            )
                            for c in list(range(1, 10))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def isvar_fourier_comparison(cls) -> SystemConfiguration:
        def get_lagmax(configuration: ISAModelConfiguration) -> str:
            return configuration.feature_computer_container_collection.feature_computer_containers[
                0
            ].file_feature_computer.bytes_computer.data_half_len  # type: ignore

        return SystemConfiguration.with_paramcomp_visres(
            "Fourier data_half_len comparison",
            get_lagmax,
            "Fourier",
            "frequency",
            [
                (
                    "isvar_fourier_comparison",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(FourierComputer(2**c)),
                                cls.get_hyperparam_classifiers(10**1, None),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(1, 10))
                        ],
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(FourierComputer(2**c)),
                                cls.get_hyperparam_classifiers(None, 10**1),
                                None,
                                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                            )
                            for c in list(range(5, 10))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def instsize_fourier_comparison(cls) -> SystemConfiguration:
        def get_lagmax(configuration: ISAModelConfiguration) -> str:
            return configuration.feature_computer_container_collection.feature_computer_containers[
                0
            ].file_feature_computer.bytes_computer.data_half_len  # type: ignore

        return SystemConfiguration.with_paramcomp_visres(
            "Fourier data_half_len comparison",
            get_lagmax,
            "Fourier",
            "frequency",
            [
                (
                    "instsize_fourier_comparison",
                    [
                        *[
                            (
                                DatasetTrain(CpuRec),
                                SimpleTest(),
                                AdaptedBytesComputer(FourierComputer(2**c)),
                                cls.get_hyperparam_classifiers(10**1, 10**1),
                                None,
                                LabelEntry.FIXED_INSTRUCTION_SIZE,
                            )
                            for c in list(range(5, 10))
                        ],
                    ],
                ),
            ],
        )

    @classmethod
    def isvar_precisions(cls) -> SystemConfiguration:
        return SystemConfiguration.with_precision_visres(
            [
                (
                    "isvar_precisions",
                    [
                        (
                            DatasetTrain(CpuRec),
                            SimpleTest(),
                            ByteDifferencePrimes(),
                            cls.get_classifiers(10**-1, 10**-2),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            SimpleTest(),
                            AdaptedBytesComputer(AutoCorrelationComputer(128)),
                            cls.get_classifiers(10**0, 10**0),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            SimpleTest(),
                            AdaptedBytesComputer(FourierComputer(32)),
                            cls.get_classifiers(10**1, 10**1),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                    ],
                ),
            ],
        )

    @classmethod
    def isvar_precisions_code(cls) -> SystemConfiguration:
        return SystemConfiguration.with_precision_visres(
            [
                (
                    "isvar_precisions_code",
                    [
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetectCode()),
                            AdaptedBytesComputer(AutoCorrelationComputer(128)),
                            cls.get_classifiers(10**0, 10**0),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetectCode()),
                            AdaptedBytesComputer(FourierComputer(32)),
                            cls.get_classifiers(10**1, 10**1),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetectCode(1500)),
                            ByteDifferencePrimes(),
                            cls.get_classifiers(10**-1, 10**-2),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                    ],
                ),
            ],
        )

    @classmethod
    def isvar_precisions_full(cls) -> SystemConfiguration:
        return SystemConfiguration.with_precision_visres(
            [
                (
                    "isvar_precisions_full",
                    [
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetect()),
                            AdaptedBytesComputer(AutoCorrelationComputer(128)),
                            cls.get_classifiers(10**0, 10**0),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetect()),
                            AdaptedBytesComputer(FourierComputer(32)),
                            cls.get_classifiers(10**1, 10**1),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            DatasetTest(IsaDetect(1500)),
                            ByteDifferencePrimes(),
                            cls.get_classifiers(10**-1, 10**-2),
                            None,
                            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                        ),
                    ],
                ),
            ],
        )

    @classmethod
    def __instsize_precisions_any(
        cls, test_mode: TestMode, identifier: str
    ) -> SystemConfiguration:
        return SystemConfiguration.with_precision_visres(
            [
                (
                    identifier,
                    [
                        (
                            DatasetTrain(CpuRec),
                            test_mode,
                            AdaptedBytesComputer(AutoCorrelationComputer(128)),
                            cls.get_hyperparam_classifiers(10**1, None),
                            None,
                            LabelEntry.INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            test_mode,
                            AdaptedBytesComputer(AutoCorrelationComputer(64)),
                            cls.get_hyperparam_classifiers(None, 10**1),
                            None,
                            LabelEntry.INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            test_mode,
                            AdaptedBytesComputer(FourierComputer(32)),
                            cls.get_hyperparam_classifiers(10**-4, 10**-4),
                            None,
                            LabelEntry.INSTRUCTION_SIZE,
                        ),
                        (
                            DatasetTrain(CpuRec),
                            test_mode,
                            AdaptedBytesComputer(AutoCorrelationComputer(2)),
                            cls.get_hyperparam_classifiers(10**1, 10**1),
                            None,
                            LabelEntry.INSTRUCTION_SIZE,
                        ),
                    ],
                ),
            ],
        )

    @classmethod
    def instsize_precisions(cls) -> SystemConfiguration:
        return cls.__instsize_precisions_any(
            SimpleTest(), "instsize_precisions"
        )

    @classmethod
    def instsize_precisions_full(cls) -> SystemConfiguration:
        return cls.__instsize_precisions_any(
            DatasetTest(IsaDetect()), "instsize_precisions_full"
        )

    @classmethod
    def instsize_precisions_code(cls) -> SystemConfiguration:
        return cls.__instsize_precisions_any(
            DatasetTest(IsaDetectCode()), "instsize_precisions_code"
        )

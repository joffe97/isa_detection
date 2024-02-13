from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from domains.feature.feature_computer_container import FeatureComputerContainer
from domains.feature.feature_computer_container_collection import (
    FeatureComputerContainerCollection,
)
from domains.feature.features_post_computers import FeaturesPostComputer, MostCommon

from domains.feature.file_feature_computer_collection import FileFeatureComputer
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.setup import Setup
from domains.system.system_modes import SystemMode
from domains.system.system_modes.test_modes import SimpleTest, CpuRecTest
from domains.system.system_modes.train_modes import ISADetectTrain
from domains.system.system import System
from domains.feature.file_feature_computers import (
    TrigramsNonZero,
    ByteFrequencyDistribution,
    Bigrams,
    EndiannessSignatures,
)


if __name__ == "__main__":
    Setup().with_all_config()
    system_mode = SystemMode(ISADetectTrain, SimpleTest)

    feature_computer_container_params: list[list] = [
        [(TrigramsNonZero, MostCommon(1000))],
        # [ByteFrequencyDistribution],
        # [Bigrams],
        # [EndiannessSignatures],
        # [EndiannessSignatures, ByteFrequencyDistribution]
    ]
    feature_computer_container_collections = [
        FeatureComputerContainerCollection(
            [
                (
                    FeatureComputerContainer(*feature)
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
        LogisticRegression(C=100_000_000, max_iter=10_000, n_jobs=-1, random_state=RANDOM_STATE),
        SVC(kernel="linear", C=1_000_000, random_state=RANDOM_STATE),
        SVC(kernel="poly", C=100_000, random_state=RANDOM_STATE),
        SVC(kernel="sigmoid", C=0.01, random_state=RANDOM_STATE),
        SVC(kernel="rbf", C=10_000, random_state=RANDOM_STATE),
        GaussianNB(),
        KNeighborsClassifier(1, n_jobs=-1),
        KNeighborsClassifier(3, n_jobs=-1),
        KNeighborsClassifier(5, n_jobs=-1),
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        # MLPClassifier(max_iter=10_000, random_state=random_state),
    ]
    files_per_architecture_list = [10]
    target_labels = ["wordsize"]

    isa_model_configuration = ISAModelConfiguration.create_every_combination(
        feature_computer_container_collections,
        classifiers,
        files_per_architecture_list,
        target_labels,
    )

    result = System(system_mode, isa_model_configuration).run_and_visualize()
    print(result)

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from domains.feature.feature_computer_collection import FeatureComputerCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes import SystemMode
from domains.system.system_modes.test_modes import SimpleTest, CpuRecTest
from domains.system.system_modes.train_modes import ISADetectTrain

from domains.system.system import System
from domains.feature.feature_computers import *


if __name__ == "__main__":
    # system_mode = system_modes.ISADetectTrainAndTest()
    system_mode = SystemMode(ISADetectTrain, SimpleTest)

    feature_computer_collections = [
        FeatureComputerCollection(feature_computers) for feature_computers in
        # [Trigrams]
        [[ByteFrequencyDistribution],
         [Bigrams],
         [EndiannessSignatures],
         [EndiannessSignatures, ByteFrequencyDistribution]]
    ]
    random_state = 42
    classifiers = [
        # BaggingClassifier(RandomForestClassifier(),
        #                   n_estimators=50, n_jobs=-1),
        RandomForestClassifier(n_jobs=-1, random_state=random_state),
        LogisticRegression(C=100_000_000, max_iter=10_000,
                           n_jobs=-1, random_state=random_state),
        SVC(kernel="linear", C=1_000_000, random_state=random_state),
        SVC(kernel="poly", C=100_000, random_state=random_state),
        SVC(kernel="sigmoid", C=0.01, random_state=random_state),
        SVC(kernel="rbf", C=10_000, random_state=random_state),
        GaussianNB(),
        KNeighborsClassifier(1, n_jobs=-1),
        KNeighborsClassifier(3, n_jobs=-1),
        KNeighborsClassifier(5, n_jobs=-1),
        DecisionTreeClassifier(random_state=random_state)
        # MLPClassifier(max_iter=10_000, random_state=random_state),
    ]
    files_per_architecture_list = [
        10
    ]
    target_labels = ["wordsize"]

    isa_model_configuration = ISAModelConfiguration.create_every_combination(
        feature_computer_collections, classifiers, files_per_architecture_list, target_labels)

    result = System(system_mode, isa_model_configuration).run()
    print(result)

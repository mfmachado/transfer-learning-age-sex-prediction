from enum import Enum


class ProblemOptimize(Enum):
    AGE = 1
    GENDER = 2


class TrainingStrategy(Enum):
    OFF_THE_SHELF = 1
    FINE_TUNING = 2
    TRAINING_FROM_SCRATCH = 3


class BrainTissue(Enum):
    GM = 1
    WM = 2
    CSF = 3
    DF = 4


class DATASETS(Enum):
    IXI = 3
    OASIS1 = 8
    OASIS2 = 9
    OASIS3 = 4
    ABIDE_I = 5
    ABIDE_II = 10
    GSP = 7
    ADNI = 11
    fcp_1000 = 12

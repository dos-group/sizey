from enum import Enum


class ERROR_STRATEGY(Enum):
    DOUBLE = 1
    MAX_EVER_OBSERVED = 2


class OFFSET_STRATEGY(Enum):
    STD = 1
    MED_UNDER = 2
    MED_ALL = 5
    STDUNDER = 6
    DYNAMIC = 7





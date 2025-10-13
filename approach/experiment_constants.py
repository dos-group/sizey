from enum import Enum


class ERROR_STRATEGY(Enum):
    """
    Defines the strategy for handling errors in the experiment.

    Attributes:
        DOUBLE (int): Strategy to double the error value.
        MAX_EVER_OBSERVED (int): Strategy to use the maximum error value ever observed.
    """
    DOUBLE = 1
    MAX_EVER_OBSERVED = 2


class OFFSET_STRATEGY(Enum):
    """
    Defines the strategy for handling offsets in the experiment.

    Attributes:
        STD (int): Strategy to use the standard deviation.
        MED_UNDER (int): Strategy to use the median of a subset of data.
        MED_ALL (int): Strategy to use the median of all data.
        STDUNDER (int): Strategy to use a modified standard deviation.
        DYNAMIC (int): Strategy to dynamically adjust the offset.
    """
    STD = 1
    MED_UNDER = 2
    MED_ALL = 5
    STDUNDER = 6
    DYNAMIC = 7





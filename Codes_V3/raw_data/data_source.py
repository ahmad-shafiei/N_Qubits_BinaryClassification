# Codes_V3/raw_data/data_source.py

from abc import ABC, abstractmethod
from .iq_container import RawIQData


class DataSource(ABC):
    """
    Abstract base class for all raw IQ data loaders
    """

    @abstractmethod
    def load(self) -> RawIQData:
        """
        Load raw IQ data and return RawIQData object
        """
        pass

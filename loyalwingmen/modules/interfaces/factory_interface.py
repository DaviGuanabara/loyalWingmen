from abc import ABCMeta, abstractmethod
# https://pypi.org/project/abcmeta/
from modules.models.drone import Drone


class IDroneFactory(metaclass=ABCMeta):

    # =================================================================================================================
    # Private
    # =================================================================================================================

    # =================================================================================================================
    # Public
    # =================================================================================================================

    @abstractmethod
    def create(self) -> Drone:
        """Abstract method."""

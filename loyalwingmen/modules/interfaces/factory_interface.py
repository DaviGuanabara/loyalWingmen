from abc import ABCMeta, abstractmethod
# https://pypi.org/project/abcmeta/


class IFactory(metaclass=ABCMeta):

    # =================================================================================================================
    # Private
    # =================================================================================================================

    # =================================================================================================================
    # Public
    # =================================================================================================================

    @abstractmethod
    def create(self):
        """Abstract method."""

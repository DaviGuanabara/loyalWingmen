import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Text
from abc import ABCMeta, abstractmethod
# https://pypi.org/project/abcmeta/
from modules.dataclasses.dataclasses import Parameters, Kinematics, Informations
from modules.control import DSLPIDControl
from dataclasses import dataclass, field
# from modules.environments.environment_models import EnvironmentParameters
from modules.dataclasses.dataclasses import EnvironmentParameters


# NÃO IMPLEMENTADO AINDA


class IEnvironment(metaclass=ABCMeta):

    def __init__(self, id: int, parameters: Parameters, kinematics: Kinematics, informations: Informations, control: DSLPIDControl, environment_parameters: EnvironmentParameters):
        self.id: int = id
        self.client_id: int = environment_parameters.client_id
        self.parameters: Parameters = parameters
        self.kinematics: Kinematics = kinematics
        self.informations: Informations = informations
        self.control: DSLPIDControl = control
        self.environment_parameters: EnvironmentParameters = environment_parameters

        self.__setup()

    # =================================================================================================================
    # Private
    # =================================================================================================================

    @abstractmethod
    def __setup(self):
        """Abstract method."""

    @abstractmethod
    def __physics(self, rpm: np.array):
        """Abstract method."""

    @abstractmethod
    def __collect_kinematics(self) -> Kinematics:
        """Abstract method."""

    # =================================================================================================================
    # Public
    # =================================================================================================================

    @abstractmethod
    def store_kinematics(self, kinematics: Kinematics):
        """Abstract method."""

    @abstractmethod
    def update_kinematics(self):
        """Abstract method."""

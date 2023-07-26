import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Text
from abc import ABCMeta, abstractmethod
# https://pypi.org/project/abcmeta/
from modules.dataclasses.dataclasses import Parameters, Kinematics, Informations, EnvironmentParameters
from modules.control.DSLPIDControl import DSLPIDControl
from dataclasses import dataclass, field
from modules.utils.enums import DroneModel


class IDrone(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, id: int, model: DroneModel, parameters: Parameters, kinematics: Kinematics, informations: Informations, control: DSLPIDControl, environment_parameters: EnvironmentParameters):
        """Abstract method."""

    # =================================================================================================================
    # Private
    # =================================================================================================================

    @abstractmethod
    def physics(self, rpm: np.ndarray):
        """Abstract method."""

    @abstractmethod
    def collect_kinematics(self) -> Kinematics:
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

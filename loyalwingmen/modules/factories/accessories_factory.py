import pybullet as p
from modules.factories.factory_models import Kinematics
import math
import numpy as np
import random
import pybullet as p


class LiDAR():
    def __init__(self, max_distance: int = 3, resolution: float = 1):
        self.max_distance = max_distance
        self.resolution = resolution

        _, self.n_theta_readings, self.n_phi_readings = self.__calculate_variables(
            max_distance, resolution)
        self.matrix: np.array = self.__gen_matrix_view(
            self.n_theta_readings, self.n_phi_readings)

    def get_flag(self, name):
        if name == "LOYALWINGMEN":
            return 1

        if name == "TARGET":
            return 2

        if name == "OBSTACLE":
            return 3

        return 0

    # ============================================================================================================
    # Setup Functions
    # ============================================================================================================

    def __calculate_variables(self, max_distance: float, resolution: float = 1):
        sphere_surface = 4 * math.pi * max_distance ** 2
        number_of_readings: int = math.ceil(resolution * sphere_surface)

        n_phi_readings: int = math.ceil(math.sqrt(number_of_readings))
        n_theta_readings: int = math.ceil(math.sqrt(number_of_readings))

        return sphere_surface, n_theta_readings, n_phi_readings

    def __gen_matrix_view(self, n_theta_readings, n_phi_readings, n_channels: int = 2):
        matrix = np.zeros((n_theta_readings, n_phi_readings, n_channels))
        return matrix

    def reset(self):
        self.matrix: np.array = self.__gen_matrix_view(
            self.n_theta_readings, self.n_phi_readings)

    # ============================================================================================================
    # Coordinates Functions
    # ============================================================================================================

    def spherical_to_cartesian(self, spherical: np.array):
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)
        radius, theta, phi
        theta is polar angle
        phi is azimuthal angle
        """

        radius, theta, phi = spherical[0], spherical[1], spherical[2]

        return [
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta)
        ]

    def cartesian_to_spherical(self, cartesian: np.array):
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)
        theta is polar angle
        phi is azimuthal angle
        """

        x, y, z = cartesian[0], cartesian[1], cartesian[2]
        x_2, y_2, z_2 = x ** 2, y ** 2, z ** 2

        return [
            math.sqrt(x_2 + y_2 + z_2),  # Ok
            math.acos(z/math.sqrt(x_2 + y_2 + z_2)),
            math.atan2(y, x)
        ]

    # ============================================================================================================
    # Matrix Functions
    # ============================================================================================================

    def __add_spherical_to_matrix(self, matrix: np.array, spherical: list, distance: float = 10, flag: int = 0):
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)
        theta is polar angle
        phi is azimuthal angle

        matrix[theta_position][phi_position]
        """

        DISTANCE_CHANNEL = 0
        FLAG_CHANNEL = 1

        if distance > self.max_distance:
            return

        n_theta_readings = len(matrix)
        n_phi_readings = len(matrix[0])
        theta_step_size = (2 * math.pi) / n_theta_readings
        phi_step_size = (2 * math.pi) / n_phi_readings

        _, theta, phi = spherical[0], spherical[1], spherical[2]
        theta_position = round(theta / theta_step_size)
        phi_position = round(phi / phi_step_size)

        if matrix[theta_position][phi_position][DISTANCE_CHANNEL] > 0 and distance > matrix[theta_position][phi_position][DISTANCE_CHANNEL]:
            return

        matrix[theta_position][phi_position][DISTANCE_CHANNEL] = distance
        matrix[theta_position][phi_position][FLAG_CHANNEL] = flag

    def __add_cartesian_to_matrix(self, matrix: np.array, cartesian: list, distance: float = 10, flag: int = 0):
        spherical = self.cartesian_to_spherical(cartesian)
        self.__add_spherical_to_matrix(matrix, spherical, distance, flag)

    def __add_end_position(self, end_position: np.array, current_position: np.array = [0, 0, 0], flag: int = 0):
        cartesian: list = end_position - current_position
        distance = np.linalg.norm(end_position - current_position)
        self.__add_cartesian_to_matrix(self.matrix, cartesian, distance, flag)

    def add_position(self, target_position: np.array = np.array([]), obstacle_position: np.array = np.array([]), loyalwingmen_position: np.array = np.array([]), current_position: np.array = np.array([0, 0, 0])):

        if len(target_position) > 0:
            self.__add_end_position(
                target_position, current_position, self.get_flag("TARGET"))

        if len(obstacle_position) > 0:
            self.__add_end_position(
                obstacle_position, current_position, self.get_flag("OBSTACLE"))

        if len(loyalwingmen_position) > 0:
            self.__add_end_position(
                loyalwingmen_position, current_position, self.get_flag("LOYALWINGMEN"))

    def get_matrix(self):
        return self.matrix

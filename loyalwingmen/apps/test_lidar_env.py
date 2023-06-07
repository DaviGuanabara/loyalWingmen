import pybullet as p
from modules.factories.drone_factory import DroneFactory
import math
import random
import numpy as np


class LiDAR_new():
    def __init__(self, max_distance: int = 3, resolution: float = 1):
        self.max_distance = max_distance
        self.resolution = resolution

        _, self.n_theta_readings, self.n_phi_readings = self.__calculate_variables(
            max_distance, resolution)
        self.matrix: np.array = self.__gen_matrix_view(
            self.n_theta_readings, self.n_phi_readings)

    # ============================================================================================================
    # Setup Functions
    # ============================================================================================================

    def __calculate_variables(self, max_distance: float, resolution: float = 1):
        sphere_surface = 4 * math.pi * max_distance ** 2
        number_of_readings: int = math.ceil(resolution * sphere_surface)

        n_phi_readings: int = math.ceil(math.sqrt(number_of_readings))
        n_theta_readings: int = math.ceil(math.sqrt(number_of_readings))

        return sphere_surface, n_theta_readings, n_phi_readings

    def __gen_matrix_view(self, n_theta_readings, n_phi_readings):
        matrix = np.zeros((n_theta_readings, n_phi_readings))
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

    def __add_spherical_to_matrix(self, matrix: np.array, spherical: list, distance: float = 10):
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)
        theta is polar angle
        phi is azimuthal angle

        matrix[theta_position][phi_position]
        """

        if distance > self.max_distance:
            return

        n_theta_readings = len(matrix)
        n_phi_readings = len(matrix[0])
        theta_step_size = (2 * math.pi) / n_theta_readings
        phi_step_size = (2 * math.pi) / n_phi_readings

        _, theta, phi = spherical[0], spherical[1], spherical[2]
        theta_position = round(theta / theta_step_size)
        phi_position = round(phi / phi_step_size)

        matrix[theta_position][phi_position] = distance

    def __add_cartesian_to_matrix(self, matrix: np.array, cartesian: list, distance: float = 10):
        spherical = self.cartesian_to_spherical(cartesian)
        self.__add_spherical_to_matrix(matrix, spherical, distance)

    def add_position(self, target_position: np.array, current_position: np.array = [0, 0, 0]):
        cartesian: list = target_position - current_position
        distance = np.linalg.norm(target_position - current_position)
        self.__add_cartesian_to_matrix(self.matrix, cartesian, distance)

    def get_matrix(self):
        return self.matrix


lidar = LiDAR_new(max_distance=5, resolution=1)


current_position = np.array([0, 0, 0])
target_position = np.array([0, 3, 3])
lidar.add_position(target_position, current_position)
print(lidar.get_matrix())


import math
import numpy as np
from typing import Tuple

# TODO colocar os hints ao redor do código


class LiDAR():
    """
        This class aims to simulate a spherical LiDAR reading. It uses physics convention for spherical coordinates (radius, theta, phi).
        In this convention, theta is the polar angle and varies between 0 and +pi, from z to xy plane.
        Phi is the azimuthal angle, and varies between -pi and +pi, from x to y.
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    def __init__(self, radius: float = 5, resolution: float = 1):
        """
            Params
            -------
            max_distance: float, the radius of the sphere
            resolution: number of sectors per m2
        """

        self.radius = radius
        self.resolution = resolution

        self.THETA_INITIAL_RADIAN = 0
        self.THETA_FINAL_RADIAN = np.pi
        self.THETA_SIZE = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN

        self.PHI_INITIAL_RADIAN = -np.pi
        self.PHI_FINAL_RADIAN = np.pi
        self.PHI_SIZE = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        self.n_theta_points, self.n_phi_points = self.__count_points(
            radius, resolution)
        self.sphere: np.array = self.__gen_sphere(
            self.n_theta_points, self.n_phi_points)

    def __get_flag(self, name):
        if name == "LOYAL_WINGMAN":
            return 1

        if name == "LOITERING_MUNITION":
            return 2

        if name == "OBSTACLE":
            return 3

        return 0

    # ============================================================================================================
    # Setup Functions
    # ============================================================================================================

    def __count_points(self, radius: float, resolution: float = 1) -> Tuple[int, int]:
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)

        This function aims to calculate the number of points in polar angle (theta) and azimuthal angle (phi), 
        in which theta is between 0 and pi and phi is between -pi and +phi.
        With 4 points as constraints, a sphere detectable sector can be obtained.
        Each sector is responsable for holding the distance and the type of the identified object. 
        The set of sectors is further translated into matrix.


        Params
        -------
        radius: float related to maximum observable distance.
        resolution: float related to number of sectors per m2.

        Returns
        -------
        n_theta_points: int
        n_phi_points: int
        """

        sector_surface: float = 1 / resolution
        sector_side: float = math.sqrt(sector_surface)

        theta_range: float = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN
        phi_range: float = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        theta_side: float = theta_range * radius
        phi_side: float = phi_range * radius

        n_theta_points: int = math.ceil(theta_side / sector_side)
        n_phi_points: int = math.ceil(phi_side / sector_side)

        return n_theta_points, n_phi_points

    def __gen_sphere(self, n_theta_points, n_phi_points, n_channels: int = 2):
        sphere = np.zeros((n_theta_points, n_phi_points, n_channels))
        return sphere

    def reset(self):
        self.sphere: np.array = self.__gen_sphere(
            self.n_theta_points, self.n_phi_points)

    # ============================================================================================================
    # Coordinates Functions
    # ============================================================================================================

    def spherical_to_cartesian(self, spherical: np.array) -> list:

        radius, theta, phi = spherical[0], spherical[1], spherical[2]
        x: float = radius * math.sin(theta) * math.cos(phi),
        y: float = radius * math.sin(theta) * math.sin(phi),
        z: float = radius * math.cos(theta)
        return [x, y, z]

    def cartesian_to_spherical(self, cartesian: np.array):

        x, y, z = cartesian[0], cartesian[1], cartesian[2]

        radius: float = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta: float = math.acos(z/radius)
        phi: float = math.atan2(y, x)

        return [radius, theta, phi]

    # ============================================================================================================
    # Matrix Functions
    # ============================================================================================================

    def __normalize_angle(self, angle, initial_angle, final_angle, n_points) -> float:
        # linear interpolation
        return round((angle - initial_angle) / (final_angle - initial_angle) * n_points) % n_points

    def __update_sphere(self, theta_point, phi_point, distance, flag):
        DISTANCE = 0
        FLAG = 1

        current_distance = self.sphere[theta_point][phi_point][DISTANCE]

        if current_distance > 0 and distance > current_distance:
            return None

        self.sphere[theta_point][phi_point][DISTANCE] = distance
        self.sphere[theta_point][phi_point][FLAG] = flag

    def __add_spherical(self, spherical: list, distance: float = 10, flag: int = 0):

        if distance > self.radius:
            return None

        _, theta, phi = spherical[0], spherical[1], spherical[2]
        theta_point = self.__normalize_angle(
            theta, self.THETA_INITIAL_RADIAN, self.THETA_FINAL_RADIAN, self.n_theta_points)
        phi_point = self.__normalize_angle(
            phi, self.PHI_INITIAL_RADIAN, self.PHI_FINAL_RADIAN, self.n_phi_points)

        self.__update_sphere(theta_point, phi_point, distance, flag)

    def __add_cartesian(self, cartesian: list, distance: float = 10, flag: int = 0):
        spherical: list = self.cartesian_to_spherical(cartesian)
        self.__add_spherical(spherical, distance, flag)

    def __add_end_position(self, end_position: np.array, current_position: np.array = [0, 0, 0], flag: int = 0):
        cartesian: list = end_position - current_position
        distance = np.linalg.norm(end_position - current_position)

        if distance > 0:
            self.__add_cartesian(cartesian, distance, flag)

    def add_position(self, loitering_munition_position: np.array = np.array([]), obstacle_position: np.array = np.array([]), loyalwingman_position: np.array = np.array([]), current_position: np.array = np.array([0, 0, 0])):

        if len(loitering_munition_position) > 0:
            self.__add_end_position(
                loitering_munition_position, current_position, self.__get_flag("LOITERING_MUNITION"))

        if len(obstacle_position) > 0:
            self.__add_end_position(
                obstacle_position, current_position, self.__get_flag("OBSTACLE"))

        if len(loyalwingman_position) > 0:
            self.__add_end_position(
                loyalwingman_position, current_position, self.__get_flag("LOYAL_WINGMAN"))

    def get_sphere(self) -> np.array:
        return self.sphere


def cartesian_to_spherical_test(lidar: LiDAR, lm_position, spherical_degree_result):
    spherical = lidar.cartesian_to_spherical(lm_position)
    radius, theta, phi = spherical[0], spherical[1], spherical[2]
    theta_degree = np.rad2deg(theta)
    phi_degree = np.rad2deg(phi)

    spherical_degree = np.array([radius, theta_degree, phi_degree])
    test_result = np.array_equal(spherical_degree, spherical_degree_result)

    print("test_result:", test_result)
    return spherical_degree


def debug(lidar: LiDAR, lm_position, spherical_degree_result):

    print("")
    print("")
    print("4 thetas com 45 graus cada um")
    print("7 phis com 45 graus cada um (o phi número 8 é igual ao de número 0: 0 = 2*pi)")
    print("polar angle (theta) is between 0 and pi")
    print("azimuthal angle (phi) is between -180 and +180.")
    print("Assim, a matriz está indo de -180 graus até 180 graus para o phi.")
    print("")

    lidar.reset()
    lidar.add_position(loitering_munition_position=np.array(lm_position))

    spherical_degree = cartesian_to_spherical_test(
        lidar, lm_position, spherical_degree_result)
    print("lm_position:", lm_position, "angular: ", spherical_degree)

    print(lidar.get_sphere())


def test():
    lidar = LiDAR(max_distance=1, resolution=1)

    debug(lidar, np.array([1, 0, 0]), np.array([1, 90, 0]))
    debug(lidar, np.array([0, 1, 0]), np.array([1, 90, 90]))
    debug(lidar, np.array([0, 0, 1]), np.array([1, 0, 0]))


# test()

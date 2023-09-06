import math
import numpy as np
from typing import Tuple, List, Dict
import pybullet as p
from gymnasium import spaces
from .sensor_interface import Sensor


from enum import Enum, auto


class Channels(Enum):
    DISTANCE_CHANNEL = auto()
    FLAG_CHANNEL = auto()


class CoordinateConverter:
    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        radius, theta, phi = spherical
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
        x, y, z = cartesian
        radius = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])


class LiDAR(Sensor):
    """
    This class aims to simulate a spherical LiDAR reading. It uses physics convention for spherical coordinates (radius, theta, phi).
    In this convention, theta is the polar angle and varies between 0 and +pi, from z to xy plane.
    Phi is the azimuthal angle, and varies between -pi and +pi, from x to y.
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    def __init__(
        self,
        parent_id: int = 0,
        client_id: int = 0,
        radius: float = 5,
        resolution: float = 1,
        debug: bool = False,
    ):
        """
        Params
        -------
        max_distance: float, the radius of the sphere
        resolution: number of sectors per m2
        """

        self.buffer = {}
        self.parent_id = parent_id
        self.client_id = client_id
        self.debug = debug
        self.debug_line_id = -1
        # print("in Lidar, debug:", self.debug)
        self.debug_lines_id = []

        self.n_channels: int = len(Channels)

        self.flag_size = 3

        self.radius = radius
        self.resolution = resolution

        self.THETA_INITIAL_RADIAN = 0
        self.THETA_FINAL_RADIAN = np.pi
        self.THETA_SIZE = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN

        self.PHI_INITIAL_RADIAN = -np.pi
        self.PHI_FINAL_RADIAN = np.pi
        self.PHI_SIZE = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        self.n_theta_points, self.n_phi_points = self.__count_points(radius, resolution)
        self.sphere: np.ndarray = self.__gen_sphere(
            self.n_theta_points, self.n_phi_points, self.n_channels
        )

    def __get_flag(self, name: str) -> float:
        if name == "LOITERING_MUNITION":
            return 0

        if name == "LOYAL_WINGMAN":
            return 0.3

        if name == "OBSTACLE":
            return 0.6

        else:
            return 1

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

        theta_side: float = theta_range  # * radius
        phi_side: float = phi_range  # * radius

        n_theta_points: int = math.ceil(theta_side / sector_side)
        n_phi_points: int = math.ceil(phi_side / sector_side)

        return n_theta_points, n_phi_points

    def convert_angle_point_to_angles(
        self, theta_point, phi_point, n_theta_points, n_phi_points
    ) -> Tuple[float, float]:
        theta_range: float = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN
        phi_range: float = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        theta = (theta_point - 0) / (
            n_theta_points - 0
        ) * theta_range + self.THETA_INITIAL_RADIAN
        phi = (phi_point - 0) / (n_phi_points - 0) * phi_range + self.PHI_INITIAL_RADIAN

        return theta, phi

    def convert_angle_point_to_cartesian(
        self,
        theta_point,
        phi_point,
        n_theta_points,
        n_phi_points,
        distance,
        current_position,
    ) -> np.ndarray:
        theta, phi = self.convert_angle_point_to_angles(
            theta_point, phi_point, n_theta_points, n_phi_points
        )
        cartesian_from_origin = CoordinateConverter.spherical_to_cartesian(
            np.array([distance * self.radius, theta, phi])
        )
        return cartesian_from_origin + current_position

    def __gen_sphere(
        self, n_theta_points, n_phi_points, n_channels: int = 2
    ) -> np.ndarray:
        return np.ones((n_channels, n_theta_points, n_phi_points), dtype=np.float32)

    def reset(self):
        self.sphere: np.ndarray = self.__gen_sphere(
            self.n_theta_points, self.n_phi_points
        )

    # ============================================================================================================
    # Matrix Functions
    # ============================================================================================================

    def __normalize_angle(self, angle, initial_angle, final_angle, n_points) -> float:
        # linear interpolation
        return (
            round((angle - initial_angle) / (final_angle - initial_angle) * n_points)
            % n_points
        )

    def __normalize_distance(self, distance, radius) -> float:
        return distance / radius

    def __update_sphere(self, theta_point, phi_point, normalized_distance, flag):
        # DISTANCE = 0
        # FLAG = 1

        current_normalized_distance = self.sphere[Channels.DISTANCE_CHANNEL.value][
            theta_point
        ][phi_point]

        if (
            current_normalized_distance > 0
            and normalized_distance > current_normalized_distance
        ):
            return None

        self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][
            phi_point
        ] = normalized_distance

        # TODO: SÓ COLOQUEI ISSO PARA FACILITAR A MUDANÇA NO NÚMERO DE CANAIS. NO CASO, EU QUERO DEIXAR EM 1.
        if self.sphere.shape[0] > 1:
            self.sphere[Channels.DISTANCE_CHANNEL.FLAG_CHANNEL.value][theta_point][
                phi_point
            ] = flag

    def __add_spherical(
        self,
        spherical: np.ndarray,
        distance: np.float32 = np.float32(10),
        flag: float = 0,
    ):
        radius = self.radius
        if distance > radius:
            return None

        _, theta, phi = spherical[0], spherical[1], spherical[2]

        normalized_distance = self.__normalize_distance(distance, radius)

        theta_point = self.__normalize_angle(
            theta,
            self.THETA_INITIAL_RADIAN,
            self.THETA_FINAL_RADIAN,
            self.n_theta_points,
        )
        phi_point = self.__normalize_angle(
            phi, self.PHI_INITIAL_RADIAN, self.PHI_FINAL_RADIAN, self.n_phi_points
        )

        self.__update_sphere(theta_point, phi_point, normalized_distance, flag)

    def __add_cartesian(
        self,
        cartesian: np.ndarray,
        distance: np.float32 = np.float32(10),
        flag: float = 0,
    ):
        spherical: np.ndarray = CoordinateConverter.cartesian_to_spherical(cartesian)
        self.__add_spherical(spherical, distance, flag)

    def __add_end_position(
        self,
        end_position: np.ndarray,
        current_position: np.ndarray = np.array([0, 0, 0]),
        flag: float = 1,
    ):
        cartesian: np.ndarray = end_position - current_position
        distance: np.float32 = np.linalg.norm(end_position - current_position)

        if distance > 0:
            self.__add_cartesian(cartesian, distance, flag)

    def debug_sphere(self, current_position):
        # print(self.sphere[self.DISTANCE_CHANNEL])
        for theta_point in range(len(self.sphere[Channels.DISTANCE_CHANNEL.value])):
            for phi_point in range(
                len(self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point])
            ):
                distance = self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][
                    phi_point
                ]
                if distance < 1:
                    # print("distance:", distance)

                    end_position = self.convert_angle_point_to_cartesian(
                        theta_point=theta_point,
                        phi_point=phi_point,
                        n_theta_points=self.n_theta_points,
                        n_phi_points=self.n_phi_points,
                        distance=distance,
                        current_position=current_position,
                    )
                    if self.debug_line_id == -1:
                        self.debug_line_id = p.addUserDebugLine(
                            lineFromXYZ=current_position,
                            lineToXYZ=end_position,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=3,
                            lifeTime=1,
                            physicsClientId=self.client_id,
                        )
                    else:
                        self.debug_line_id = p.addUserDebugLine(
                            lineFromXYZ=current_position,
                            lineToXYZ=end_position,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=3,
                            lifeTime=1,
                            replaceItemUniqueId=self.debug_line_id,
                            physicsClientId=self.client_id,
                        )

    def add_position(
        self,
        loitering_munition_position: np.ndarray = np.array([]),
        obstacle_position: np.ndarray = np.array([]),
        loyalwingman_position: np.ndarray = np.array([]),
        current_position: np.ndarray = np.array([0, 0, 0]),
    ):
        if len(loitering_munition_position) > 0:
            self.__add_end_position(
                loitering_munition_position,
                current_position,
                self.__get_flag("LOITERING_MUNITION"),
            )

        if len(obstacle_position) > 0:
            self.__add_end_position(
                obstacle_position, current_position, self.__get_flag("OBSTACLE")
            )

        if len(loyalwingman_position) > 0:
            self.__add_end_position(
                loyalwingman_position,
                current_position,
                self.__get_flag("LOYAL_WINGMAN"),
            )

        if self.debug:
            self.debug_sphere(current_position)

    def get_sphere(self) -> np.ndarray:
        return self.sphere

    def read_data(self) -> Dict:
        return {"lidar": self.sphere}

    def update_data(self):
        pass
    
    def buffer_flight_state_data(self, message: Dict, publisher_id: int):
        self.buffer[str(publisher_id)] = message
    #TODO: lidar tem que se adptar ao padrão sensor.
    #TODO: import class QuadcopterType(Enum): from quadcopter.py
    #QUADCOPTER = auto()
    #LOYALWINGMAN = auto()
    #LOITERINGMUNITION = auto()


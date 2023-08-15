import math
import numpy as np
from typing import Tuple, List
import pybullet as p
from gymnasium import spaces

from enum import Enum

class Channels(Enum):
    DISTANCE_CHANNEL = 0
    FLAG_CHANNEL = 1
    
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


class LiDAR:
    """
    This class aims to simulate a spherical LiDAR reading. It uses physics convention for spherical coordinates (radius, theta, phi).
    In this convention, theta is the polar angle and varies between 0 and +pi, from z to xy plane.
    Phi is the azimuthal angle, and varies between -pi and +pi, from x to y.
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    def __init__(self, radius: float = 5, resolution: float = 1, client_id: int = 0, debug: bool = False):
        """
        Params
        -------
        max_distance: float, the radius of the sphere
        resolution: number of sectors per m2
        """
        self.client_id = client_id
        self.debug = debug
        
        #print("in Lidar, debug:", self.debug)
        self.debug_lines_id = []

        #TODO: fazer um enun com os channels, e atualizar lá no demo.env
        self.n_channels: int = 2
        
        self.flag_size = 3

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

        theta_side: float = theta_range * radius
        phi_side: float = phi_range * radius

        n_theta_points: int = math.ceil(theta_side / sector_side)
        n_phi_points: int = math.ceil(phi_side / sector_side)

        return n_theta_points, n_phi_points
    
    def convert_angle_point_to_angles(self, theta_point, phi_point, n_theta_points, n_phi_points) -> Tuple[float, float]:
        
        theta_range: float = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN
        phi_range: float = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN
        
        theta = (theta_point - 0) / (n_theta_points - 0) * theta_range + self.THETA_INITIAL_RADIAN
        phi = (phi_point - 0) / (n_phi_points - 0) * phi_range + self.PHI_INITIAL_RADIAN
        
        
        return theta, phi
    
    def convert_angle_point_to_cartesian(self, theta_point, phi_point, n_theta_points, n_phi_points, distance, current_position) -> np.ndarray:
        theta, phi = self.convert_angle_point_to_angles(theta_point, phi_point, n_theta_points, n_phi_points)
        cartesian_from_origin = CoordinateConverter.spherical_to_cartesian(np.array([distance * self.radius, theta, phi]))
        return cartesian_from_origin + current_position
  

    def __gen_sphere(self, n_theta_points, n_phi_points, n_channels: int = 2) -> np.ndarray:
        # it is assumed the following shape: CxHxW (channels first)
        sphere = np.ones((n_channels, n_theta_points,
                          n_phi_points), dtype=np.float32)
        return sphere

    def reset(self):
        self.sphere: np.ndarray = self.__gen_sphere(
            self.n_theta_points, self.n_phi_points
        )
        
        if self.debug:
            for line_id in self.debug_lines_id:
                p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
            self.debug_lines_id = []
            


    # ============================================================================================================
    # Matrix Functions
    # ============================================================================================================

    def __normalize_angle(self, angle, initial_angle, final_angle, n_points) -> float:
        # linear interpolation
        return (
            round((angle - initial_angle) /
                  (final_angle - initial_angle) * n_points)
            % n_points
        )

    def __normalize_distance(self, distance, radius) -> float:
        return distance / radius

    def __update_sphere(self, theta_point, phi_point, normalized_distance, flag):
        #DISTANCE = 0
        #FLAG = 1

        current_normalized_distance = self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][phi_point]

        if (
            current_normalized_distance > 0
            and normalized_distance > current_normalized_distance
        ):
            return None

        self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][phi_point] = normalized_distance
        self.sphere[Channels.DISTANCE_CHANNEL.FLAG_CHANNEL.value][theta_point][phi_point] = flag

    def __add_spherical(self, spherical: np.ndarray, distance: np.float32 = np.float32(10), flag: float = 0):
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

    def __add_cartesian(self, cartesian: np.ndarray, distance: np.float32 = np.float32(10), flag: float = 0):
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
        #print(self.sphere[self.DISTANCE_CHANNEL])
        for theta_point in range(len(self.sphere[Channels.DISTANCE_CHANNEL.value])):
            for phi_point in range(len(self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point])):
                
                distance = self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][phi_point]
                if distance < 1:
                    #print("distance:", distance)
                    end_position = self.convert_angle_point_to_cartesian(theta_point=theta_point, phi_point=phi_point, n_theta_points=self.n_theta_points, n_phi_points=self.n_phi_points, distance=distance, current_position=current_position)
                    line_id = p.addUserDebugLine(lineFromXYZ = current_position, lineToXYZ = end_position, lineColorRGB = [1, 0, 0], lineWidth = 3, lifeTime = 0.1, physicsClientId=self.client_id)
                    self.debug_lines_id.append(line_id)
                    
        return
    
    
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
                obstacle_position, current_position, self.__get_flag(
                    "OBSTACLE")
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

    # ============================================================================================================
    # Observation Functions
    # ============================================================================================================

    def observation_space(self) -> spaces.Box:
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray

        """
        # a workaround to work with gymnasium
        sphere: np.ndarray = self.sphere
        return spaces.Box(
            low=0,
            high=1,
            shape=sphere.shape,
            dtype=np.float32,
        )
    
    def get_features(self) -> List:
        """
        Find elements below one in a 3-dimensional Lidar observation and return a list of tuples.

        This function iterates through a 3-dimensional Lidar observation represented as a nested list
        and identifies the elements whose values are below 1. It returns a list of tuples, where each
        tuple contains the channel, theta, phi indices, and the value of the element.

        Params
        -------
        lidar_observation: list of lists of lists
            The 3-dimensional Lidar observation.

        Returns
        -------
        list
            A list of tuples (channel, theta, phi, value) for elements below one.
        """
    
        below_one_list = []

        for channel in range(len(self.sphere)):
            
            for theta in range(len(self.sphere[channel])):
                for phi in range(len(self.sphere[channel][theta])):
                    value = self.sphere[channel][theta][phi]
                    if value < 1:
                        below_one_list.append((channel, theta, phi, value))
        
        return below_one_list
    
    def parameters(self) -> dict:
        parameters = {}
        
        parameters["n_channels"] = self.n_channels
        parameters["n_theta_points"] = self.n_theta_points
        parameters["n_phi_points"] = self.n_phi_points
        
        parameters["radius"] = self.radius
        parameters["resolution"] = self.resolution
        
        return parameters
                


# ============================================================================================================
# Test
# ============================================================================================================


def cartesian_to_spherical_test(lidar: LiDAR, lm_position, spherical_degree_result):
    spherical = CoordinateConverter.cartesian_to_spherical(lm_position)
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
    print(
        "7 phis com 45 graus cada um (o phi número 8 é igual ao de número 0: 0 = 2*pi)"
    )
    print("polar angle (theta) is between 0 and pi")
    print("azimuthal angle (phi) is between -180 and +180.")
    print("Assim, a matriz está indo de -180 graus até 180 graus para o phi.")
    print("")

    lidar.reset()
    lidar.add_position(loitering_munition_position=np.array(lm_position))

    spherical_degree = cartesian_to_spherical_test(
        lidar, lm_position, spherical_degree_result
    )
    print("lm_position:", lm_position, "angular: ", spherical_degree)

    print(lidar.get_sphere())


def test():
    lidar = LiDAR(radius=1, resolution=.1)

    debug(lidar, np.array([1, 0, 0]), np.array([1, 90, 0]))
    debug(lidar, np.array([0, 1, 0]), np.array([1, 90, 90]))
    debug(lidar, np.array([0, 0, 1]), np.array([1, 0, 0]))
#test()

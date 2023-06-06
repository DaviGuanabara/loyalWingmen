import pybullet as p
from modules.factories.factory_models import Drone
from modules.factories.factory_models import Kinematics
import math


class LiDAR():

    def __init__(self, client_id, gadget, radius: int = 10, resolution: float = 4):
        assert hasattr(gadget, 'kinematics') and isinstance(
            gadget.kinematics, Kinematics), "LiDAR cannot operate on object without kinematic attribute"
        self.client_id = client_id
        self.gadget = gadget
        self.radius = radius
        # (readings per sphere_surface) (readings per square meter) (the drone has 0.6 meter, so at least it has to be 2 readings)
        self.resolution = resolution

        self.__calculate_variables__()

    def __calculate_variables__(self):
        self.sphere_surface = 4 * math.pi * self.radius ** 2
        number_of_readings: int = math.ceil(
            self.resolution * self.sphere_surface)

        self.n_phi_readings: int = math.ceil(math.sqrt(number_of_readings))
        self.n_theta_readings: int = math.ceil(math.sqrt(number_of_readings))

    def gen_polar_coordinates(self):
        gadget_kinematics: Kinematics = self.gadget.kinematics

        position = gadget_kinematics.position
        angular_position = gadget_kinematics.angular_position

        radius = self.radius
        phi_step_size = (2 * math.pi) / self.n_phi_readings
        theta_step_size = (2 * math.pi) / self.n_theta_readings

        polar_coordinates = []
        for i_phi in range(self.n_x_readings):
            for i_theta in range(self.n_theta_readings):
                phi = i_phi * phi_step_size
                theta = i_theta * theta_step_size

                polar_coordinates.append[(radius, theta, phi)]

        return polar_coordinates

    def polar2cart(self, radius, theta, phi):
        return [
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta)
        ]

    def polar_to_cartesian(polar_coordinate=(0, 0, 0)):
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        # https://stackoverflow.com/questions/48348953/spherical-polar-co-ordinate-to-cartesian-co-ordinate-conversion
        pass

    def polar_to_cartesian_batch(polar_coordinates: list = []):

        for i in range(len(polar_coordinates)):
            polar_coordinates

    def read(self):

        rayTestBatch

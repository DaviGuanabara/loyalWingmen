import pybullet as p
from modules.factories.factory_models import Kinematics
import math
import numpy as np
import random
import pybullet as p


class LiDAR():

    def __init__(self, client_id, gadget, initial_radius: int = .8, radius: int = 3, resolution: float = 1):
        assert hasattr(gadget, 'kinematics') and isinstance(
            gadget.kinematics, Kinematics), "LiDAR cannot operate on object without kinematic attribute"
        self.client_id = client_id
        self.gadget = gadget
        self.radius = radius
        self.initial_radius = initial_radius
        # (readings per sphere_surface) (readings per square meter) (the drone has 0.6 meter, so at least it has to be 2 readings)
        self.resolution = resolution
        self.lines = []
        self.single_line = None
        self.__calculate_variables__()

    def __calculate_variables__(self):
        self.sphere_surface = 4 * math.pi * self.radius ** 2
        number_of_readings: int = math.ceil(
            self.resolution * self.sphere_surface)

        self.n_phi_readings: int = math.ceil(math.sqrt(number_of_readings))
        self.n_theta_readings: int = math.ceil(math.sqrt(number_of_readings))

    def gen_polar_batch(self, radius: int = 10):

        #radius = self.radius
        phi_step_size = (2 * math.pi) / self.n_phi_readings
        theta_step_size = (2 * math.pi) / self.n_theta_readings

        polar_coordinates = []
        for i_phi in range(self.n_phi_readings):
            for i_theta in range(self.n_theta_readings):
                phi = i_phi * phi_step_size
                theta = i_theta * theta_step_size

                polar_coordinates.append([radius, theta, phi])

        return polar_coordinates

    def polar_to_cartesian(self, polar: list = [0, 0, 0]):
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        # https://stackoverflow.com/questions/48348953/spherical-polar-co-ordinate-to-cartesian-co-ordinate-conversion
        radius = polar[0]
        theta = polar[1]
        phi = polar[2]

        return [
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta)
        ]

    def polar_batch_to_cartesian_batch(self, polar_batch: list = [[0, 0, 0]]):

        cartesian_batch = [self.polar_to_cartesian(
            polar) for polar in polar_batch]

        return cartesian_batch

    def calculate_initial_points(self):

        gadget_kinematics: Kinematics = self.gadget.kinematics

        gadget_position = gadget_kinematics.position

        polar_batch = self.gen_polar_batch(radius=self.initial_radius)
        cartesian_batch = self.polar_batch_to_cartesian_batch(polar_batch)

        initial_points = [np.add(cartesian, gadget_position)
                      for cartesian in cartesian_batch]

        return initial_points
    

    def calculate_end_points(self):
        gadget_kinematics: Kinematics = self.gadget.kinematics

        gadget_position = gadget_kinematics.position

        polar_batch = self.gen_polar_batch(radius=self.radius)
        cartesian_batch = self.polar_batch_to_cartesian_batch(polar_batch)

        end_points = [np.add(cartesian, gadget_position)
                      for cartesian in cartesian_batch]

        return end_points

    def add_debug_lines(self, initial_points, end_points):
        columns = []
        for initial_point in initial_points:

            rows = []
            for end_point in end_points:
            
                if len(self.lines) > 0:
                    rows.append(p.addUserDebugLine(lineFromXYZ=initial_point, lineToXYZ=end_point, replaceItemUniqueId=self.lines[initial_point][end_point], physicsClientId=self.client_id))
                else:
                    rows.append(p.addUserDebugLine(lineFromXYZ=initial_point, lineToXYZ=end_point, physicsClientId=self.client_id))
            columns.append(rows)

        self.lines = columns


    def add_single_debug_lines(self, initial_points, end_points):

        
        i = random.choice(range(len(initial_points)))
        initial_point = initial_points[i]
        end_point = end_points[i]


        if self.single_line is not None:
            self.single_line = p.addUserDebugLine(lineFromXYZ=initial_point, lineToXYZ=end_point, replaceItemUniqueId=self.single_line, physicsClientId=self.client_id)

        else:
            self.single_line = p.addUserDebugLine(lineFromXYZ=initial_point, lineToXYZ=end_point, physicsClientId=self.client_id) 


    def read(self):

        initial_points = self.calculate_initial_points()
        end_points = self.calculate_end_points()

        self.add_single_debug_lines(initial_points, end_points)

        return p.rayTestBatch(rayFromPositions=initial_points,
                       rayToPositions=end_points, physicsClientId=self.client_id)
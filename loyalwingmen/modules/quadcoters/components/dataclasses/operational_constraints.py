import numpy as np
from dataclasses import dataclass


@dataclass(order=True)
class OperationalConstraints:
    speed_limit: float = 0
    angular_speed_limit: float = 0
    acceleration_limit: float = 0
    angular_acceleration_limit: float = 0

    weight: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_z_torque: float = 0
    hover_rpm: float = 0
    gnd_eff_h_clip: float = 0
    max_xy_torque: float = 0


'''

OperationalConstraints:
speed_limit: The maximum linear speed the quadcopter is allowed to have.
angular_speed_limit: The maximum angular speed (rotational speed) the quadcopter is allowed to have.
acceleration_limit: The maximum linear acceleration the quadcopter is allowed to have.
angular_acceleration_limit: The maximum angular acceleration the quadcopter is allowed to have.
weight: The weight of the quadcopter, same as the WEIGHT in QuadcopterSpecs (seems redundant).
max_rpm: The maximum revolutions per minute of the quadcopter's motors.
max_thrust: The maximum thrust the quadcopter can produce.
max_z_torque: The maximum torque the quadcopter can produce about the Z-axis.
hover_rpm: The revolutions per minute of the motors when the quadcopter is hovering (maintaining a steady altitude).
gnd_eff_h_clip: Related to the ground effect. Perhaps the height below which the quadcopter starts experiencing a significant ground effect.
max_xy_torque: The maximum torque the quadcopter can produce about the X or Y axes.

'''
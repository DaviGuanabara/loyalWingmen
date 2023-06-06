import pybullet as p
from modules.factories.drone_factory import DroneFactory
#from modules.decorators.drone_decorator import DroneDecorator


p.connect(p.DIRECT)
x_drone = DroneFactory().gen_extended_drone()

print(x_drone.get_LiDAR_readings())
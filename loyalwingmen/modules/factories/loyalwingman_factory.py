from modules.factories.drone_factory import DroneFactory
from modules.models.loyalwingman import LoyalWingman, Drone


class LoyalWingmanFactory(DroneFactory):
    def __init__(self):
        super().__init__()

    def create(self) -> Drone:

        id, model, parameters, informations, kinematics, control, environment_parameters = \
            super().load_drone_attributes()

        loyalwingman = LoyalWingman(id, model, parameters, informations, kinematics,
                                    control, environment_parameters)

        return loyalwingman

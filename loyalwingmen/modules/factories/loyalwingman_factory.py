from modules.factories.drone_factory import DroneFactory
from modules.models.loyalwingman import LoyalWingman, Drone


class LoyalWingmanFactory(DroneFactory):
    def __init__(self, environment_parameters):
        super().__init__(environment_parameters)

    def create(self) -> Drone:
        (
            id,
            model,
            parameters,
            informations,
            kinematics,
            control,
            environment_parameters,
            lidar,
        ) = super().load_drone_attributes()

        loyalwingman = LoyalWingman(
            id=id,
            model=model,
            parameters=parameters,
            kinematics=kinematics,
            informations=informations,
            control=control,
            environment_parameters=environment_parameters,
            lidar=lidar,
        )

        return loyalwingman

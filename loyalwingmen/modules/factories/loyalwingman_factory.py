from modules.factories.drone_factory import DroneFactory
from modules.models.loyalwingman import LoyalWingman, Drone


class LoyalWingmanFactory(DroneFactory):
    def __init__(self, environment_parameters, speed_amplification: float = 1, debug: bool = False):
    
        super().__init__(environment_parameters, speed_amplification=speed_amplification, debug=debug)

    def create(self) -> LoyalWingman:
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

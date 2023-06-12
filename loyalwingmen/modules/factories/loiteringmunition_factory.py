from modules.factories.drone_factory import DroneFactory
from modules.models.loiteringmunition import LoiteringMunition, Drone


class LoiteringMunitionFactory(DroneFactory):
    def __init__(self):
        super().__init__()

    def create(self) -> Drone:

        id, model, parameters, informations, kinematics, control, environment_parameters = \
            super().load_drone_attributes()
        loiteringmunition = LoiteringMunition(id=id, model=model, parameters=parameters, kinematics=kinematics,
                                              informations=informations, control=control, environment_parameters=environment_parameters)

        return loiteringmunition

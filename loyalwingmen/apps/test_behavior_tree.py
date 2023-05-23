from modules.behaviors.tree.drone_behavior_tree import StandartAgentBehaviorTree
from modules.factories.drone_factory import AgentFactory
import numpy as np
from pathlib import Path
import os

base_path = str(Path(os.getcwd()).parent.absolute())
path = base_path + "\\" + "assets\\" + "cf2x.urdf"

agent_factory = AgentFactory()

quadcopter = agent_factory.gen_drone(
    client_id=-1,
    urdf_file_path=path,  # "assets/" + "cf2x.urdf",  # drone_model.value + ".urdf"
    initial_position=np.array([0, 0, 0]),
    gravity_acceleration=9.8,
)

# behavior_tree = StandartAgentBehaviorTree(quadcopter)
# print(behavior_tree.update())

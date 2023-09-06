import numpy as np
import pybullet as p
import platform
import os
from pathlib import Path

import xml.etree.ElementTree as etxml
from typing import Tuple

from modules.models.sensors.lidar_sensor import LiDAR


from modules.models.drones.loyalwingman import LoyalWingman
from modules.models.drones.loiteringmunition import LoiteringMunition

from modules.control.DSLPIDControl import DSLPIDControl
from modules.utils.enums import DroneModel

from modules.models.drones.drone import (
    Drone,
    Parameters,
    Kinematics,
    Informations,
    EnvironmentParameters,
)

from enum import Enum, auto

##AINDA ESTOU EDITANDO O DRONE_FACTORY, COM A AJUDA DO CHAT GPT

class DroneType(Enum):
    LOYALWINGMAN = auto()
    LOITERINGMUNITION = auto()

class DroneURDFHandler:
    def __init__(self, drone_model: DroneModel, environment_parameters: EnvironmentParameters):
        self.environment_parameters = environment_parameters
        
        self.urdf_file_path = DroneURDFHandler.create_path(drone_model=drone_model)
        self.tree = etxml.parse(self.urdf_file_path)
        self.root = self.tree.getroot()
    
    @staticmethod
    def create_path(drone_model: DroneModel) -> str:
        urdf_name = drone_model.value + ".urdf"
        base_path = str(Path(os.getcwd()).parent.absolute())
        if platform.system() == "Windows":
            return base_path + "\\" + "assets\\" + urdf_name
        else:
            return base_path + "/" + "assets/" + urdf_name
        
    def load_to_pybullet(self, initial_position, initial_quaternion):
        return p.loadURDF(
            self.urdf_file_path,
            initial_position,
            initial_quaternion,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.environment_parameters.client_id
        )    
        
    def load_parameters(self):
        """Loads parameters from an URDF file.
        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """

        URDF_TREE = self.root
        M = float(URDF_TREE[1][0][1].attrib["value"])
        L = float(URDF_TREE[0].attrib["arm"])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])
        IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
        IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
        IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib["kf"])
        KM = float(URDF_TREE[0].attrib["km"])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
        PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
        DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
        DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])
        return Parameters(
            M=M,
            L=L,
            THRUST2WEIGHT_RATIO=THRUST2WEIGHT_RATIO,
            J=J,
            J_INV=J_INV,
            KF=KF,
            KM=KM,
            COLLISION_H=COLLISION_H,
            COLLISION_R=COLLISION_R,
            COLLISION_Z_OFFSET=COLLISION_Z_OFFSET,
            MAX_SPEED_KMH=MAX_SPEED_KMH,
            GND_EFF_COEFF=GND_EFF_COEFF,
            PROP_RADIUS=PROP_RADIUS,
            DRAG_COEFF=DRAG_COEFF,
            DW_COEFF_1=DW_COEFF_1,
            DW_COEFF_2=DW_COEFF_2,
            DW_COEFF_3=DW_COEFF_3,
        )
    

class DroneFactory():
    def __init__(
        self,
        environment_parameters: EnvironmentParameters,
        drone_model: DroneModel = DroneModel.CF2X,
    ):
        
        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug
        
        self.drone_model = drone_model
        self.drone_urdf_handler = DroneURDFHandler(self.drone_model, self.environment_parameters)
        self.environment_parameters = environment_parameters
        
        self.lidar = None
        
        
        
    # =================================================================================================================
    # Set
    # =================================================================================================================
    

    def set_initial_position(self, initial_position: np.ndarray):
        self.initial_position = initial_position

    def set_initial_angular_position(self, initial_angular_position: np.ndarray):
        self.initial_angular_position = initial_angular_position
        self.initial_quaternion = p.getQuaternionFromEuler(initial_angular_position)

    def set_LiDAR(self, lidar: LiDAR):
        self.lidar = lidar

    # =================================================================================================================
    # Compute
    # =================================================================================================================


    def __compute_kinematics(self) -> Kinematics:
        return Kinematics(
            position=self.initial_position,
            angular_position=self.initial_angular_position,
        )

    def __compute_parameters(self):
        return self.drone_urdf_handler.load_parameters()

    def __compute_informations(self, parameters: Parameters):
       
        gravity_acceleration = self.environment_parameters.G
        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1
        
        L = parameters.L
        M = parameters.M
        KF = parameters.KF
        KM = parameters.KM
        PROP_RADIUS = parameters.PROP_RADIUS
        GND_EFF_COEFF = parameters.GND_EFF_COEFF
        THRUST2WEIGHT_RATIO = parameters.THRUST2WEIGHT_RATIO

        gravity = gravity_acceleration * M
        max_rpm = np.sqrt((THRUST2WEIGHT_RATIO * gravity) / (4 * KF))
        max_thrust = 4 * KF * max_rpm**2
        max_z_torque = 2 * KM * max_rpm**2
        hover_rpm = np.sqrt(gravity / (4 * KF))
        speed_limit = VELOCITY_LIMITER * parameters.MAX_SPEED_KMH * KMH_TO_MS
        gnd_eff_h_clip = (
            0.25
            * PROP_RADIUS
            * np.sqrt((15 * max_rpm**2 * KF * GND_EFF_COEFF) / max_thrust)
        )
        max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(
            2
        ) 

        informations = Informations()
        informations.gravity = gravity
        informations.max_rpm = max_rpm
        informations.max_thrust = max_thrust
        informations.max_z_torque = max_z_torque
        informations.hover_rpm = hover_rpm
        informations.speed_limit = speed_limit
        informations.gnd_eff_h_clip = gnd_eff_h_clip
        informations.max_xy_torque = max_xy_torque

        return informations

    def __compute_drone_model(self):
        return self.drone_model  # DroneModel.CF2X

    def __compute_control(
        self,
        droneParameters: Parameters,

    ) -> DSLPIDControl:
        return DSLPIDControl(
            self.drone_model, droneParameters, self.environment_parameters
        )

    

    ################### create ###############################

    def load_drone_attributes(
        self
    ) -> Tuple[
        int,
        DroneModel,
        Parameters,
        Informations,
        Kinematics,
        DSLPIDControl,
        EnvironmentParameters,
    ]:
        id = self.drone_urdf_handler.load_to_pybullet(self.initial_position, self.initial_quaternion)
        model = self.__compute_drone_model()
        parameters = self.__compute_parameters()
        informations = self.__compute_informations(parameters)
        kinematics = self.__compute_kinematics()
        control = self.__compute_control(
            parameters, 
        )
        environment_parameters = self.environment_parameters
        

        return (
            id,
            model,
            parameters,
            informations,
            kinematics,
            control,
            environment_parameters,
        )

    def create(self, type: DroneType, position: np.ndarray, ang_position: np.ndarray) -> Drone:
        
        self.set_initial_position(position)
        self.set_initial_angular_position(ang_position)
        
        attributes = self.load_drone_attributes()
        
        constructor = {
            DroneType.LOYALWINGMAN: LoyalWingman,
            DroneType.LOITERINGMUNITION: LoiteringMunition
        }.get(type, Drone)
        
        drone = constructor(*attributes)
        drone.set_lidar(self.lidar)
        return drone
    
    


    """
        Compute Information, em drone factory, é usado para computar informações importantes
        a cerca do drone, como o peso, o raio das helices, etc.
        Aqui há duas informações muito importantes, o speed_limit e o velocity_amplification.
        (talvez o nome melhor fosse velocity_amplifier).
        
        O primeiro, speed_limit, é o limite de velocidade que o drone pode atingir, ou seja,
        se o drone atingir esse limite, ele não pode mais acelerar.
        Como a ação é um vetor de 3 dimensões, que varia de 0 até 1, o limite de velocidade é multiplicado 
        nesse vetor, com 1 sendo a velocidade máxima em um eixo, e 0 a velocidade mínima.
        
        O cáculo do speed_limit é dado por:
        speed_limit = VELOCITY_LIMITER * parameters.MAX_SPEED_KMH * KMH_TO_MS,
        no qual parameters.MAX_SPEED_KMH é oriundo do arquivo urdf, e KMH_TO_MS é uma constante.
        Velocity_limiter é uma constante que é definida na própria função.
        O debug mostra que speed_limit = 8,333333333333334 m/s, ou seja, 30 km/h.
        
        Inicialmente, a rede neural retorna uma ação (vetor velocidade) baixa, em torno de 0.08 (como isso é proporcional
        pode-se pensar em 8% da velocidade máxima), o que faz com que o drone acelere lentamente, algo em torno de 
        0.64 m/s. Para uma frequência de 15 hz, isso daria em torno de 0.04 m/s por ação. Dado que a recompensa
        é a distancia entre o drone e o alvo, isso daria em uma variação na recompensa de 0.04. Caso o drone esteja
        a 5 m de distancia do alvo, a recompensa por não fazer nada é de 5 +- 0.04, ou seja, 4.96 a 5.04, o que pode
        ser descartado devido as aproximações.
        
        Há dois caminhos a se seguir: 
        1. Diminuir a frequencia da rl.
        2. Aumentar a velocidade do drone.
        
        1. Diminuir a frequencia da rl.
        A frequência da RL é dado por rl_frequency e é definida como argumento do environment.
        Apesar de aumentar o impacto das ações sob a recompensa, diminuir a frequencia da rl 
        aumenta o tempo de treinamento, afinal se antes a rl fazia 1000 ações por segundo
        e agora faz 100, vai demorar 10x mais para treinar o mesmo número de passos de tempo.
        Na realidade, não é uma variação linear assim. A diminuição da frequencia da rl aumenta o impacto do OverHead do
        Reset do Environment, que é o tempo que o ambiente leva para resetar, e isso aumenta o tempo de treinamento final, 
        pois o mesmo passará por mais Resets dentro da mesma faixa de passos de tempo.
        
        
        2. Aumentar a velocidade do drone.
        Há duas constantes, até que redundantes, criadas para construir a velocidade do drone.
        Antes de mais nada, a ação tomada pela rede neural é um vetor velocidade que varia de -1 até 1 nos
        eixos x, y e z. Assim, trata-se de um grau de proporcionalidade e sentido, assim 1 é equivalente a 
        100% da velocidade máxima, e -1 a 100% da velocidade máxima em sentido contrário ao crescimento do eixo.
        
        As variáveis são: Speed Limit e Velocity Amplification.
        a Target_Velocity, a velocidade resultante, é dada pela multiplicação da Speed Limit e Velocity Amplification.
        Aumentar a velocidade significaria aumentar a velocity amplification, algo que impacta diretamente na recompensa, mas também
        no comportamento do drone. Aumentar a velocidade do drone significa que ele vai se mover mais rápido, e assim
        irrealisticamente e a RL com baixa frequencia pode não ser capaz de controlá-lo.
        
        Velocidade do Drone X frequencia da rl
        
        Por conta do impacto no tempo, a redução da frequencia da rl não é uma solução viavel, então a solução é aumentar a amplificação,
        ou ao menos encontrar algo que balanceie a amplificação com a frequencia da rl.
        
        Dentro das configurações atuais, um episódio com 10 segundos, velocity_amplification em 1, e a frequencia da rl em 1hz,
        para treinar 1 milhão de passos de tempo, é necessário 2:09:20 horas de treinamento (em uma média de 186 it/s).
        
        Aumentar a frequencia do drone pode ser corrigido com o aumento da velocidade
        
        Como contraste, ampliar a frequencia da rl para 15hz e aumentar a velocity amplification para 15, para treinar 1 milhão de passos de tempo (1,199,994),
        demorou 1:16:53 horas de treinamento (em uma média de 322 it/s), uma redução de 53 minutos.
        
        
        É importante notar que isso tudo é considerando a física inativa. Com a física ativada, precisaremos considerar o tempo de transição
        entre o drone estar parado até o mesmo alcançar a velocidade desejada e essa ser convertida em impacto na recompensa.
        
        De forma simplificada, surgem algumas perguntas: 
        1. Qual é o valor mínimo de variação da recompensa que faça com que a rede neural a perceba ?
        2. Qual é o tempo mínimo de variação da recompensa devemos esperar para que o acumulado da variação da recompensa seja perceptível ?
        3. Qual é a velocidade mínima que o drone deve poder atingir mantendo um impacto perceptível na recompensa ?
        4. Qual é a velocidade máxima que o drone deve poder atingir ?
        5. Como equilibrar Speed Limit e Velocity Amplification X frequencia da rl ?
        5.1. Devo igualar o Velocity Amplification e a frequencia da RL ?
        A frequencia está sendo variada em baysian_optimizer -> suggest_params
        
        
        Proposta
        Uma solução possível para a recompensa seria relacioná-la não à distância ao alvo, mas sim a taxa de variação dessa distância.
        Para um alvo estático, isso pode ser bastante vantajoso, pois o drone não precisa se mover muito para manter a recompensa. O problema 
        é que é uma recompensa atrasada, pois o drone precisa efetivamente se mover para perceber a variação na recompensa.
        
        Talvez também possa modular o derivativo com o proporcinal, assim a recompensa seria a distância ao alvo + a taxa de variação da distância ao alvo. Porém
        adicionaria duas novas variáveis para tunar, o que pode ser um problema.
        
        TODO:
        Esses parâmetros talvez devem ser difinidos no Environment e não de forma estática aqui.
        """
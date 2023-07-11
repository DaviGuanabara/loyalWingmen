import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Dict, List, Optional, Tuple, Type, Union


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


# TODO: tenho que separar o CNN do NN.

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
#model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4",
#            policy_kwargs=policy_kwargs, verbose=1)
#model.learn(1000)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, activation='ReLU'):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            getattr(nn, activation)(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            getattr(nn, activation)(),
            nn.Flatten(),

        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param hidden_dims: (list) number of units for each hidden layer
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list = [64, 64],
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = self.create_mlp(
            feature_dim, hidden_dims, last_layer_dim_pi)

        # Value network
        self.value_net = self.create_mlp(
            feature_dim, hidden_dims, last_layer_dim_vf)

    def create_mlp(self, input_dim, hidden_dims, output_dim):
        """
        Creates a MLP with the specified dimensions.

        :param input_dim: (int) The dimension of the input layer
        :param hidden_dims: (list) A list of integers where each integer is the dimension of a hidden layer
        :param output_dim: (int) The dimension of the output layer
        :return: a MLP as a nn.Sequential instance
        """
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.hidden_dims = [32, 32]
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.hidden_dims)


# model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
# model.learn(5000)

"""

class CustomNN(nn.Module):

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.

        O código então utiliza a função th.no_grad() para desativar 
        a necessidade de calcular gradientes para as operações realizadas 
        dentro de seu bloco. Isso economiza memória e é útil quando 
        você precisa executar operações com tensores, mas não precisa dos gradientes.


    def __init__(self, observation_space: spaces.Box, hidden_dims: list = [64, 64], activation='ReLU'):

        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        print(observation_space)
        # TODO: observation_space está entrando como um número. e agora ?
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            getattr(nn, activation)(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            getattr(nn, activation)(),
            nn.Flatten(),

        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        self.linear = self.create_mlp(n_flatten, hidden_dims, activation)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = hidden_dims[-1]
        self.latent_dim_vf = hidden_dims[-1]

        neural_network = nn.Sequential(self.cnn, self.linear)

        # Policy network
        self.policy_net = neural_network
        # Value network
        self.value_net = neural_network

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:


        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        
        

        print("foward")
        print(observation.shape)
        return self.forward_actor(observation), self.forward_critic(observation)

    def forward_actor(self, observation: th.Tensor) -> th.Tensor:
        return self.policy_net(observation)

    def forward_critic(self, observation: th.Tensor) -> th.Tensor:
        return self.value_net(observation)

    def create_mlp(self, input_dim, hidden_dims=[64, 64], activation='ReLU'):
        

        Cria uma MLP de dimensões dinâmicas.

        :param input_dim: (int) Dimensão da entrada.
        :param output_dim: (int) Dimensão da saída.
        :param hidden_dims: (tuple) Dimensões das camadas ocultas.
        :param activation: (str) Função de ativação a ser usada nas camadas ocultas.
        :return: Uma MLP como uma instância de nn.Module.
   
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(getattr(nn, activation)())

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            modules.append(getattr(nn, activation)())

        return nn.Sequential(*modules)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        print("custom actor está sendo inicializado")
        print(kwargs)
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        print(kwargs)
        self.hidden_dims = kwargs["net_arch"]["pi"]
        self.observation_space = observation_space

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        print(self.observation_space)

        self.mlp_extractor = CustomNN(
            self.observation_space, self.hidden_dims, activation='ReLU')


# model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
# model.learn(5000)
"""

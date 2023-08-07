import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from gymnasium import spaces
import numpy as np
from typing import Tuple


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

        net_arch = kwargs.get('net_arch')
        default_hiddens = [256, 256, 256]

        if net_arch is None:
            warnings.warn(f"net_arch is None. Using default hiddens: {default_hiddens}")
            self.hiddens = default_hiddens
        elif not isinstance(net_arch, dict):
            warnings.warn(f"net_arch is not a dictionary. Using default hiddens: {default_hiddens}")
            self.hiddens = default_hiddens
        else:
            # Get the 'pi' key from the dictionary, if it exists
            self.hiddens = net_arch.get('pi', default_hiddens)

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
        self.mlp_extractor = CustomNetwork(self.features_dim, self.hiddens)

    def __str__(self):
        return f"CustomActorCriticPolicy (hidden layers: {self.hiddens})"
    
    

#TODO: Add Observation_Space. Add ObservationType.
class MixObservationNN(nn.Module):
    def __init__(self, matrix_input_shape: Tuple[int, int, int], tuple_input_shape: Tuple[int, ...], features_dim: int = 256):
        super(MixObservationNN, self).__init__()

        # Feature extractor for the matrix observation (using Conv2d layers)
        self.matrix_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=matrix_input_shape[0], out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Feature extractor for the tuple observation (using fully connected layers)
        self.tuple_feature_extractor = nn.Sequential(
            nn.Linear(int(np.prod(tuple_input_shape)), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Compute shape by doing one forward pass for the matrix feature extractor
        with torch.no_grad():
            n_flatten_matrix = self.matrix_feature_extractor(
                torch.rand(1, *matrix_input_shape)).shape[1]

        # Compute shape by doing one forward pass for the tuple feature extractor
        with torch.no_grad():
            n_flatten_tuple = self.tuple_feature_extractor(
                torch.rand(1, *tuple_input_shape)).shape[1]

        # Concatenate both feature extractors and pass through a linear layer
        self.concatenated_dim = n_flatten_matrix + n_flatten_tuple
        self.final_layer = nn.Sequential(
            nn.Linear(self.concatenated_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, matrix_obs: torch.Tensor, tuple_obs: torch.Tensor) -> torch.Tensor:
        matrix_features = self.matrix_feature_extractor(matrix_obs)
        tuple_features = self.tuple_feature_extractor(tuple_obs.flatten(start_dim=1))
        concatenated_features = torch.cat((matrix_features, tuple_features), dim=1)
        return self.final_layer(concatenated_features)
    
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    """Actor of the Actor-Critic Method
    """

    def __init__(self, learning_rate:float=0.0001, fc1_dims:int=400,
        fc2_dims:int=300, input_dims:np.ndarray=np.array([48]),
        n_actions:int=8) -> None:
        """Constructor

        Args:
            learning_rate (float, optional): Learning rate for Adam optimizer.
                Defaults to 0.0001.
            fc1_dims (int, optional): Size of the first fully connected hidden
                layer. Defaults to 400.
            fc2_dims (int, optional): Size of the second fully connected hidden
                layer. Defaults to 300.
            input_dims (np.ndarray, optional): Dimension of the input. Defaults
                to np.array([48]).
            n_actions (int, optional): Number of actions to predict. Defaults to
                8.
        """
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)

    def forward(self, observation:T.float) -> T.float:
        """Pass Input through Neural Net

        Args:
            observation (T.float): Observation received by the environment.

        Returns:
            T.float: Action values.
        """
        x = self.fc1(observation)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = T.tanh(self.mu(x))

        return x

class Critic(nn.Module):
    """Critic of the Actor-Critic Method
    """

    def __init__(self, learning_rate:float=0.001, fc1_dims:int=400,
        fc2_dims:int=300, input_dims:np.ndarray=np.array([48]),
        n_actions:int=8) -> None:
        """Constructor

        Args:
            learning_rate (float, optional): Learning rate for Adam optimizer.
                Defaults to 0.001.
            fc1_dims (int, optional): Size of the first fully connected hidden
                layer. Defaults to 400.
            fc2_dims (int, optional): Size of the second fully connected hidden
                layer. Defaults to 300.
            input_dims (np.ndarray, optional): Dimension of the input. Defaults
                to np.array([48]).
            n_actions (int, optional): Number of actions to predict. Defaults to
                8.
        """
        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.0003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)

    def forward(self, observation:T.float, action:T.float) -> T.float:
        """Pass Input through Neural Net

        Args:
            observation (T.float): Observation received by the environment.
            action (T.float): Action received by the actor.

        Returns:
            T.float: Q-value indication the value of the action.
        """
        state_value = self.fc1(observation)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
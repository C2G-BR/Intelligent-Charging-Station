from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import os
import torch

from .ddpg import Actor, Critic
from .noise import OrnsteinUhlenbeckActionNoise
from .replay_buffer import ReplayBuffer

class Agent(ABC):
    """Abstract class from that actuals agents will inherit 
    """

    @abstractmethod
    def predict(self,
        observation:list[tuple[bool, float, float, float, int]],
        max_power:float
    ) -> list[float]:
        """Abstract method 

        Args:
            observation (list[tuple[bool, float, float, float, int]]): 
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            max_power (float):
                Maximum power of the charging station in watts

        Returns:
            list[float]: List with power that each slot receives
        """
        pass

class FCFSAgent(Agent):
    """First come first served agent; inherits from Agent
    """
    def __init__(self) -> None:
        """Constructor
        """
        pass

    def predict(self,
        observation:list[tuple[bool, float, float, float, int]],
        max_power:float
    ) -> list[float]: 
        """Algorithm that calculates the charging power for each slot.

        The algorithm proceeds according to the 'first come first served'
        principle. 

        The vehicle that has been parked in a slot for the longest time receives
        the most power. The remaining power is distributed to the other vehicles
        (which are still sorted by how long they have been parked in the
        charging slot) until there is no more power left to distribute.  

        The observation contains the vehicle states from the different
        positions.
        
        Args:
            observation (list[tuple[bool, float, float, float, int]]):
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            max_power (float):
                Maximum power of the charging station in watts

        Returns:
            list[float]: List with power that each slot receives
        """

        idxs = list(range(len(observation)))
        tmp = zip(observation, idxs)

        # sort vehicles according to dwell time
        vehicles_sorted = sorted(tmp, key = lambda x: x[0][5], reverse=True)
        
        actions = [0] * len(observation)

        # iterate through all slots
        for vehicle, idx in vehicles_sorted:
            if not (vehicle[1]/vehicle[2] >= 1) and not (vehicle[4] == -1):
                if max_power > vehicle[4]:
                    # charge current vehicle with maximum amount of power
                    actions[idx] = vehicle[4]
                    max_power -= vehicle[4]
                else:
                    # if a slot is empty no power will be removed
                    actions[idx] = max_power
                    max_power = 0

        return actions

class DDPGAgent(Agent):
    """Reinforcment learning (Deep Learning Agent) ;inherits from Agent

    DDPG: Deep Deterministic Policy Gradient
    """

    def __init__(
        self,
        pi_lr:float=0.001,
        q_lr:float=0.001,
        gamma:float=0.99,
        batch_size:int=64,
        noise:OrnsteinUhlenbeckActionNoise=None,
        min_replay_size:int=256,
        replay_buffer_size:int=100000,
        tau:float=0.005,
        input_dims:list[int]=[48],
        n_actions:int=8,
        init_epsilon:float=0.99,
        epsilon_decay:float=0.997,
        epsilon_min:float=0.05,
    ) -> None:
        """Constructor

        Within the constructor the models (Actor & Critic) and their target
        models are initialized. Further all necessary parameters for training
        will be set.

        Epsilon is ultimately decisive for whether the calculated action by the
        agent is determined by the models or by random. Over time, Epsilon
        decreases, hence the models are used with more frequency. 

        Args:
            pi_lr (float, optional):
                Learning rate determines how much to change the actor model in
                response to the calculated error every time the model weights
                are being updated. Defaults to 0.001.
            q_lr (float, optional):
                Learning rate determines how much to change the critic model in
                response to the calculated error every time the model weights
                are being updated. Defaults to 0.001.
            gamma (float, optional):
                Discount factor for future rewards. Defaults to 0.99.
            batch_size (int, optional):
                Equals the number of training examples in one forward/backward
                pass. Defaults to 64.
            noise (OrnsteinUhlenbeckActionNoise, optional):
                Using parameter noise, agents can learn their tasks faster than.
                Defaults to None.
            min_replay_size (int, optional):
                The minimum number of experiences kept. Defaults to 256.
            replay_buffer_size (int, optional):
                Corresponds to the size of the replay buffer. This contains
                historical observations. Defaults to 100000.
            tau (float, optional):
                Determines how fast the target models will be updated.
                Defaults to 0.005.
            input_dims (list[int], optional):
                Corresponds to the input size of the model. Technically, this it
                refers to the current state of the environment in vector format.
                Defaults to [48].
            n_actions (int, optional):
                Corresponds to the number of positions within the charging
                station for which the model should predict actions. Technically,
                this translates to the number of neurons in the output layer.
                Defaults to 8.
            init_epsilon (float, optional):
                Initial value for epsilon. Defaults to 0.99.
            epsilon_decay (float, optional):
                Factor by which Epsilon reduces. Defaults to 0.997.
            epsilon_min (float, optional):
                Lower epsilon limit, from which epsilon must not change anymore.
                Defaults to 0.05.
        """

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available()
            else 'cpu')

        # Neural networks
        # Policy Network
        self.pi = Actor(
            learning_rate=pi_lr,
            input_dims=input_dims,
            n_actions=n_actions
        ).to(self.device)
        
        self.target_pi = Actor(
            learning_rate=pi_lr,
            input_dims=input_dims,
            n_actions=n_actions
        ).to(self.device)
        
        self.pi_optimizer = self.pi.optimizer

        # Evaluation Network
        self.q = Critic(
            learning_rate=q_lr,
            input_dims=input_dims,
            n_actions=n_actions).to(self.device)
        self.target_q = Critic(
            learning_rate=q_lr,
            input_dims=input_dims,
            n_actions=n_actions).to(self.device)
        self.q_optimizer = self.q.optimizer


        self.sync_weights()

        self.noise = noise

        # Replay buffer
        self.min_replay_size = min_replay_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Training
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Constants
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions

    @staticmethod
    def prepare_observation(
        observation:list[tuple[bool, float, float, float, float, int]]
    ) -> np.array:
        """Prepare observations and converts it to numpy.array

        Transformation includes flatten process which is necessary to pass it as
        an input to the model. 
        Number of observations equals batch_size.

        Args:
            observation (list[tuple[bool, float, float, float, float, int]]):
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)

        Returns:
            np.array: new format for observations which can be handled by the
                model
        """
        observation = np.array(observation).flatten()
        return observation

    @staticmethod
    def normalize_actions(actions:np.array, max_power:float) -> np.array:
        """Normalizes actions according to the maximum amount of charging power

        Args:
            actions (np.array): predicted actions by the agent
            max_power (float): maximum amount of charging power through the
            charging station

        Returns:
            np.array: normalized actions
        """
        # Normalize the actions and calculate the absolute share based on the
        # percentage share.
        sum_actions = np.sum(actions) 
        normalized_actions = (actions/sum_actions) * \
            max_power if sum_actions > 0 else actions
        return normalized_actions

    def predict(
        self,
        observation:list[tuple[bool, float, float, float, float, int]]
    ) -> np.array:
        """Predicts new actions

        Actions will range between -1 and 1. Before those actions will be fed
        as input into the environment, the actions must be normalized according
        to the charging station's maximum power.

        Args:
            observation (list[tuple[bool, float, float, float, float, int]]):
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)

        Returns:
            np.array: actions that corresponds to output layer
        """
        with torch.no_grad():
            # Prepare observations -> convert it to a matching format
            obs = DDPGAgent.prepare_observation(observation)
            obs = torch.from_numpy(obs).type(torch.float).to(self.device)
            # transforms input to final shape
            obs = obs.view((-1, *obs.shape))
            action = self.pi(obs)[0]
            action = action.detach().cpu().numpy()
        return action

    def random_action(self) -> torch.FloatTensor:
        """Generates random action based on uniform distribution

        Returns:
            torch.FloatTensor: Vector with random actions
        """
        return torch.FloatTensor(self.n_actions).uniform_(-1,1)

    def train(self) -> Dict[str, torch.tensor]:
        """Trains the agent for one iteration

        Trains the agent and returns the loss.

        Returns:
            Dict[str, torch.tensor]:
                [key]: `critic_loss` or `actor_loss`
                [value]: numeric loss
        """
        # Loss statistics
        loss_results = {}
        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs, actions, rewards, new_obs, done = sample['observation'], \
            sample['action'], \
            sample['reward'], \
            sample['new_observation'], \
            sample['done']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device) \
            .view((-1,1))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(dtype=torch.float,
            device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device) \
            .view((-1,1))

        # set models to evalution mode
        self.target_pi.eval()
        self.target_q.eval()
        self.q.eval()

        # Train q network
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - done) * self.target_q(new_obs,
                self.target_pi(new_obs))
        predicted = self.q(obs, actions)
        loss = ((targets - predicted) ** 2).mean()
        loss_results['critic_loss'] = loss.data

        self.q_optimizer.zero_grad()
        self.q.train()
        loss.backward()
        self.q_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = False

        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs = sample['observation']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        self.q.eval()
        self.pi.eval()

        # Train pi network
        predicted = self.q(obs, self.pi(obs))
        loss = -predicted.mean()
        loss_results['actor_loss'] = loss.data
        self.pi_optimizer.zero_grad()
        self.pi.train()
        loss.backward()
        self.pi_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = True

        return loss_results

    def update_eps(self) -> None:
        """Update Epsilon with Decay

        Epsilon is decaying over time to reduce the randomness until it reaches
        a limit. This method should be executed after every episode during
        training.
        """
        if self.epsilon > self.epsilon_min: # check if limit is exceeded
            self.epsilon *= self.epsilon_decay # reduce epsilon
        else:
            self.epsilon = self.epsilon_min

    def predict_eps(
        self,
        observation:list[tuple[bool, float, float, float, float, int]]
    ) -> np.array:
        """Determines agents new action during training

        This function checks if a random number is grater than epsilon. If this
        is the case, the new action is calculated randomly. Otherwise, the model
        predicts a new action.

        Args:
            observation (list[tuple[bool, float, float, float, float, int]]):
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)

        Returns:
            np.array: Array with actions
        """
        with torch.no_grad():                
            random = np.random.rand()
            if random < self.epsilon:
                action = self.random_action()
                action = action.numpy()
            else:
                action = self.predict(observation) + self.noise()
                action = np.clip(action, 0, 1.0)

            return action

    def add_experience(
        self,
        observation:list[tuple[bool, float, float, float, float, int]],
        action:np.array,
        reward:float,
        new_observation:list[tuple[bool, float, float, float, float, int]],
        done:bool
    ) -> None:
        """Add new observation to experience collection

        Args:
            observation (list[tuple[bool, float, float, float, float, int]]):
                Previous observation:
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            action (np.ndarray):
                Action computed on the basis of the previous observation
            reward (float):
                Reward for performing the current action.
            new_observation
                (list[tuple[bool, float, float, float, float, int]]):
                Observation obtained after performing the predicted action:
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            done (bool):
                True if the agent has reached the goal with his action.
        """
        observation = DDPGAgent.prepare_observation(observation)
        new_observation = DDPGAgent.prepare_observation(new_observation)
        self.replay_buffer.record(
            observation, 
            action,
            reward,
            new_observation,
            done
        )

    def update(self) -> None:
        """Updates the weights of the target models
        """
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(),
                self.target_pi.parameters()):

                target_pi_param.data = (1.0 - self.tau) * \
                    target_pi_param.data + self.tau * pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(),
                self.target_q.parameters()):

                target_q_param.data = (1.0 - self.tau) * target_q_param.data + \
                    self.tau * q_param.data

    def sync_weights(self) -> None:
        """Synchronizes the weights of the models with the corresponding target
        models
        """
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(),
                self.target_pi.parameters()):
                target_pi_param.data = pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(),
                self.target_q.parameters()):
                target_q_param.data = q_param.data

    def save_agent(self, save_path:str) -> None:
        """Saves the agent

        Stores the weights of the four models from the agent locally.

        Args:
            save_path (str): folder name where models should be stored
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        torch.save(self.target_pi.state_dict(), target_pi_path)

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        torch.save(self.target_q.state_dict(), target_q_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        torch.save(self.pi.state_dict(), pi_path)

        q_path = os.path.join(save_path, "q_network.pth")
        torch.save(self.q.state_dict(), q_path)

    def load_agent(self, save_path:str) -> None:
        """Loads previously trained models from file

        Loads the weights of:
            - actor
            - target actor
            - critic
            - target critic
        After loading the model, the weights will be synchronized.

        Args:
            save_path (str): folder name where models are stored
        """
    
        pi_path = os.path.join(save_path, "pi_network.pth")
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pi.load_state_dict(torch.load(pi_path,
                                           map_location=map_location))
        self.pi.eval()

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        self.target_pi.load_state_dict(torch.load(target_pi_path,
                                                  map_location=map_location))
        self.target_pi.eval()

        q_path = os.path.join(save_path, "q_network.pth")
        self.q.load_state_dict(torch.load(q_path,
                                          map_location=map_location))
        self.q.eval()

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        self.target_q.load_state_dict(torch.load(target_q_path,
                                                 map_location=map_location))
        self.target_q.eval()

        self.sync_weights()

    def __str__(self) -> str:
        """Returns the name of agent

        Returns:
            str: name of agent
        """
        return str(self.pi) + str(self.q)
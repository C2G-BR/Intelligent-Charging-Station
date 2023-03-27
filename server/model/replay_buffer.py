from __future__ import annotations
from typing import Dict

import numpy as np

from collections import deque
from random import choices

class ReplayBuffer(object):
    """Replay Buffer containing previous Experiences"""

    def __init__(self, capacity:int):
        """Constructor

        Args:
            capacity (int): Size of the replay buffer. When a deque is full, 
            appending new elements drops a corresponding number of elements at
            the end.
        """

        self.queue = deque(maxlen=capacity)

    def record(
        self,
        obs:list[tuple[bool, float, float, float, float, int]],
        action:np.array,
        reward:float,
        new_obs:list[tuple[bool, float, float, float, float, int]],
        done:bool
    ) -> None:
        """Appends new observation to experience

        Args:
            obs (list[tuple[bool, float, float, float, float, int]]):
                Previous observation:
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            action (np.array):
                Action computed on the basis of the previous observation.
            reward (float): Reward for performing the current action.
            new_obs (list[tuple[bool, float, float, float, float, int]]):
                Observation obtained after performing the predicted action:
                List consisting of n tuples, where n corresponds to the number
                of slots of the complete charging station.
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
            done (bool): True if the agent has reached the goal with his action.
        """
        entry = (obs, action, reward, new_obs, done)
        self.queue.append(entry)

    def get_batch(self, batch_size:int) -> Dict[str, np.array]:
        """Returns batch with previous experiences

        Returns a random batch containing previous experiences with the size of
        the batch_size.

        Args:
            batch_size (int): Specifies the size of the batch

        Returns:
            Dict[str, np.array]:
            ['observation]: old observation from experience
            ['action']:
                recently performed actions for corresponding observations
            ['reward']: reward after performing corresponding actions
            ['new_observation']:
                observations after performing corresponding actions
            ['done']:
                was the overall objective achieved with the corresponding action
        """
        sample = choices(self.queue, k=batch_size)
        out_dict = {
            'observation':[],
            'action':[],
            'reward':[],
            'new_observation':[],
            'done':[]
        }
        for o, a, r, o2, d in sample:
            out_dict['observation'].append(o)
            out_dict['action'].append(a)
            out_dict['reward'].append(r)
            out_dict['new_observation'].append(o2)
            out_dict['done'].append(d)

        for k, v in out_dict.items():
            out_dict[k] = np.array(v)

        return out_dict

    def size(self) -> int:
        """Returns the size of experiences

        Returns:
            int: Length of experience queue
        """
        return len(self.queue)
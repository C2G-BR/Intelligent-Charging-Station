from __future__ import annotations

import math
import numpy as np

class Vehicle():
    """Vehicle class
    """

    def __init__(self,
        position:int,
        capacity_max:float=0,
        charging_power_min:float=0,
        charging_power_max:float=10,
        power_dissipation:float=0,
        limit:float= 0.8,
        exp_charge:float=-0.05,
        time_steps:int=0,
        capacity_history:list[float]=[]
    ) -> None:
        """Constructor

        Args:
            position (int):
                Slot position on which the vehicle is parked/charged.
            capacity_max (float, optional):
                Maximum amount of battery capacity in kwh. Defaults to 0.
            charging_power_min (float, optional):
                Minimum amount of power in kw that is required to charge the
                vehicle. Everything below will be ignored. Defaults to 0.
            charging_power_max (float, optional): 
                Maxmimum amount of power in kw that is required to charge the
                vehicle. Everything above will be throttled.
                'charging_power_min' should be smaller than
                'charging_power_max'. Defaults to 10.
            power_dissipation (float, optional):
                Loss of power during charing due to heat etc. Value should range
                between 0 and 1. Defaults to 0.
            limit (float, optional):
                Limit from when the vehicle is evaluated as fully charged.
                Defaults to 0.8.
            exp_charge (float, optional):
                Describes in the exponential function how very strong/weak the
                loading is. Should be negative. Defaults to -0.05.
            time_steps (int, optional):
                Indicates how many time steps the vehicle stands on a slot.
                Defaults to 0.
            capacity_history (list[float], optional):
                Capacity history containing new capacity values after each
                charging step. Defaults to [].
        """

        self.position = position
        self.capacity_max = capacity_max
        self.charging_power_min = charging_power_min
        self.charging_power_max = charging_power_max
        self.power_dissipation = power_dissipation
        self.capacity_history = capacity_history

        self.time_steps = time_steps
        self.limit = limit
        self.exp_charge = exp_charge
        self.exceeds_limit = self.exceeds_reward_limit()

    def charge(self,
        charging_power:float,
        duration_time_step:float
    ) -> float:
        """Charges the vehicle

        The vehicle is charged with a certain power for a certain time.
        The formula for this is:

            new_capacity = max_capacity -
                    (max_capacity - current_capacity) *
                    e^(charging_exponent * charging_power * charging_duration) 
            
        The power loss is subtracted from the gained capacity.        

        Furthermore, a vehicle has a power consumption interval. Inserted power
        below the consumption interval are ignored and charged power that
        exceeds the consumption interval are reduced to the maximum possible
        charging power limit.

        Moreover, the time steps how long the vehicle is already on the slot is
        increased.

        Args:
            charging_power (float): charging power in kw
            duration_time_step (float): Duration in h how long the vehicle
                should be charged with the given power.

        Returns:
            float: Difference in capacity during charging step
                (new_capacity - old_capacity).
        """

        self.time_steps += 1

        if self.charging_power_min > charging_power:
            charging_power = 0
        elif self.charging_power_max < charging_power:
            charging_power = self.charging_power_max

        new_capacity = self.capacity_max - \
            (self.capacity_max - self.capacity_current) * \
            math.exp(self.exp_charge * charging_power * duration_time_step) 

        diff_capacity = (new_capacity - self.capacity_current) * \
            (1 - self.power_dissipation)

        new_capacity = diff_capacity + self.capacity_current

        if new_capacity > self.capacity_max:
            diff_capacity = self.capacity_max - self.capacity_current
            new_capacity = self.capacity_max

        self.capacity_history.append(new_capacity)

        # Add 10% of max capacity as reward if vehicle is charged over 80%
        # within this time step.
        if not self.exceeds_limit and self.exceeds_reward_limit():
            diff_capacity += self.capacity_max * 0.1
            self.exceeds_limit = True

        return diff_capacity

    def exceeds_reward_limit(self) -> bool:
        """Checks if the vehicle is sufficiently charged

        The vehicle should not be charged to 100% in order to save the battery.
        As soon as the limit (default 80%) is reached, True is returned.

        Returns:
            bool: capacity limit is exceeded
        """
        return self.capacity_in_percentage() > self.limit

    def capacity_in_percentage(self) -> float:
        """Calculates current capacity in percentage

        Returns:
            float: current capacity between 0 and 1
        """
        return self.capacity_current / self.capacity_max

    def convert_to_tuple(self) -> tuple[bool, float, float, float, float, int]:
        """Converts vehicle to a tuple

        If there is a vehicle on the slot, its state is returned as a tuple.

        Returns:
            tuple[bool, float, float, float, float, int]:
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
        """

        return (
            True,
            self.capacity_current,
            self.capacity_max,
            self.charging_power_min,
            self.charging_power_max,
            self.time_steps
        )

    @property
    def capacity_current(self) -> float:
        """Returns current capacity from vehicle

        Reads out and returns the last entry from capacity history.

        Returns:
            float: current capacity
        """
        return self.capacity_history[-1]

    @classmethod
    def generate_vehicle(cls, position:int) -> Vehicle:
        """Generates a random vehicle based on predefined profiles

        Based on a random number, a random vehicle is created. The number
        equals the index within the predefined profiles from which we receive
        the relevant attributes needed to initialize the object. 

        Args:
            position (int): slot position on which the vehicle is located

        Returns:
            Vehicle: random vehicle object
        """

        capacity_history = [[0], [5], [20], [32], [20], [28]]
        capacity_max = [100, 60, 80, 95, 25, 100]
        charging_power_min = [1.4, 1.6, 1.3, 1.4, 1.5, 1.6]
        charging_power_max = [200, 150, 50, 160, 70, 190]
        power_dissipation = [0.05, 0.1, 0.034, 0.052, 0.054, 0.045]

        idx = np.random.choice(list(range(len(capacity_history))))

        return cls(
            position=position,
            capacity_history=capacity_history[idx],
            capacity_max=capacity_max[idx],
            charging_power_min=charging_power_min[idx],
            charging_power_max=charging_power_max[idx],
            power_dissipation=power_dissipation[idx]
        )

    @classmethod
    def generate_vehicle_from_json(cls, data:dict) -> Vehicle:
        """Generates a vehicle from a dict/json object

        This function allows to quickly create a vehicle object from data from
        the frontend.

        Args:
            data (dict): dict containing all necessary data to create a vehicle

        Returns:
            Vehicle: vehicle object
        """
        
        return cls(
            position=data['position'],
            capacity_max=data['capacity_max'],
            charging_power_min=data['charging_power_min'],
            charging_power_max=data['charging_power_max'],
            power_dissipation=data['power_dissipation'],
            capacity_history=data['capacity_history'],
            limit=data['limit'],
            time_steps=data['time_steps'],
            exp_charge=data['exp_charge']
        )

    
    @classmethod
    def empty_space(cls) -> tuple[bool, float, float, float, int]:
        """Returns default values for an slot without a vehicle

        If there is no vehicle on a slot, a tuple with default values is
        returned for the observation. Since the values are wrong in content and
        do not make sense, the agent should later learn and understand that
        there is no vehicle on the slot with these values.

        Returns:
            tuple[bool, float, float, float, float, int]:
                [0]: if charging slot is in usage
                [1]: current capacity
                [2]: max capacity
                [3]: min charging power
                [4]: max charging power
                [5]: time steps (how many time steps the slot is usage)
        """
        return (False, -1, -1, -1, -1, -1)
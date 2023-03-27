from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .vehicle import Vehicle

class Environment():
    """Environment for the Intelligent Charging Station
    """
    def __init__(self, max_power:float, max_steps:int=200, positions:int=8,
        duration_time_step:float=1, is_train:bool=True, seed:int=0) -> None:
        """Constructor

        Args:
            max_power (float): Maximum amount of electric power in kW the agent
                is able to assign.
            max_steps (int, optional): Maximum number of steps taken each
                episode. Defaults to 200.
            positions (int, optional): Number of lots available for charging.
                Defaults to 8.
            duration_time_step (float, optional): Time in h that will be
                simulated each time step. Defaults to 1.
            is_train (bool, optional): Whether the environment runs in training
                mode. This includes the random generation of cars for training.
                Defaults to True.
            seed (int, optional): Seed used for random generators.
        """
        self.max_power = max_power
        self.max_steps = max_steps
        self.current_step = 0
        self.duration_time_step = duration_time_step
        self.positions = positions
        self.is_train = is_train

        np.random.seed(seed)

    def get_available_positions(self) -> list[int]:
        """Retrieve IDs of available Charging Positions

        Returns:
            list[int]: IDs of available positions.
        """
        reserved_positions = [vehicle.position for vehicle in self.vehicles]

        all_positions = set(range(self.positions))

        return list(all_positions - set(reserved_positions))

    def get_observation(self) -> list[
        tuple[bool, float, float, float, float, int]]:
        """Return the current vehicle state in a processable format

        Returns:
            list[tuple[bool, float, float, float, float, int]]: list containing
                tuples:
                [0]: Whether there is a vehicle on the position.
                [1]: Current capacity of the vehicle in kWh.
                [2]: Maximum capacity of the vehicle in kWh.
                [3]: Minimum charging power required by the vehicle in kW.
                [4]: Maximum charging power supported by the vehicle in kW.
                [5]: Number of time steps the vehicle did not leave the
                    position.
        """
        observation = []

        available_positions = self.get_available_positions()

        for p in range(self.positions):
            if p in available_positions:
                observation.append(Vehicle.empty_space())
            else:
                vehicle = self.get_vehicle_by_position(p)
                observation.append(vehicle.convert_to_tuple())
                
        return observation

    def get_vehicle_by_position(self, position:int) -> Vehicle:
        """Retrieve Vehicle at specified Position

        Args:
            position (int): Position of the vehicle to be returned.

        Raises:
            Exception: If no vehicle is found at the given position.

        Returns:
            Vehicle: Vehicle at the given position.
        """
        for vehicle in self.vehicles:
            if vehicle.position == position:
                return vehicle
        
        raise Exception('No vehicle found at position {position}.')

    def reset(self) -> tuple[list[tuple[bool, float, float, float, float, int]],
        float, bool]:
        """Resets the environment

        Resets the environment to random state. This method should be called
        during training only.

        Returns:
            tuple[list[tuple[bool, float, float, float, float, int]], float,
                bool]:
                [0]: list containing tuples:
                    [0]: Whether there is a vehicle on the position.
                    [1]: Current capacity of the vehicle in kWh.
                    [2]: Maximum capacity of the vehicle in kWh.
                    [3]: Minimum charging power required by the vehicle in kW.
                    [4]: Maximum charging power supported by the vehicle in kW.
                    [5]: Number of time steps the vehicle did not leave the
                        position.
                [1]: Reward
                [2]: Whether the epoch is done.
        """
        self.current_step = 0
        self.vehicles = []
        for i in range(self.positions):
            new_vehicle = np.random.choice([True, False], p=[0.5, 0.5])
            if new_vehicle:
                self.vehicles.append(Vehicle.generate_vehicle(i))
        
        return self.get_observation(), 0, False

    def step(self, charging_power_per_vehicle:list[float]) -> tuple[
        list[tuple[bool, float, float, float, float, int]], float, bool]:
        """Execute one Time Step

        Args:
            charging_power_per_vehicle (list[float]): Charging power assigned to
                each position. The indices will be used as the positions. For
                positions which do not contain a car or it should not be
                charged, add 0 to the list.

        Returns:
            tuple[list[tuple[bool, float, float, float, float, int]], float,
                bool]:
                [0]: list containing tuples:
                    [0]: Whether there is a vehicle on the position.
                    [1]: Current capacity of the vehicle in kWh.
                    [2]: Maximum capacity of the vehicle in kWh.
                    [3]: Minimum charging power required by the vehicle in kW.
                    [4]: Maximum charging power supported by the vehicle in kW.
                    [5]: Number of time steps the vehicle did not leave the
                        position.
                [1]: Reward
                [2]: Whether the epoch is done.
        """ 
        acc_reward = 0
        available_positions = []
        self.current_step += 1

        for vehicle in self.vehicles:
            charging_power = charging_power_per_vehicle[vehicle.position]
            acc_reward += vehicle.charge(charging_power,
                duration_time_step=self.duration_time_step)

        if self.is_train:
            available_positions = self.get_available_positions()
            for p in range(self.positions):
                if p in available_positions:
                    new_vehicle = np.random.choice([True, False],
                        p=[0.15, 0.85])
                    if new_vehicle:
                        self.vehicles.append(
                            Vehicle.generate_vehicle(position=p))
                else:
                    vehicle_leaves = np.random.choice([True, False],
                        p=[0.1, 0.9])
                    if vehicle_leaves:
                        self.vehicles.remove(self.get_vehicle_by_position(p))
        
        done = self.current_step >= self.max_steps

        return self.get_observation(), acc_reward, done

    def show(self) -> None:
        """Display the current charging state in a bar chart
        """
        plt.figure(1)
        plt.clf()

        heights = []
        available_positions = self.get_available_positions()
        
        # Retrive the current values for all positions.
        for p in range(self.positions):
            if p in available_positions:
                heights.append(0)
            else:
                vehicle = self.get_vehicle_by_position(position=p)
                heights.append(vehicle.capacity_in_percentage()*100)

        plt.bar(list(range(self.positions)), heights)
        plt.pause(1)

    @classmethod
    def generate_env_from_json(cls, data:dict) -> Environment:
        """Create an Environment from JSON Input

        Args:
            data (dict): JSON-style parameters used within the environment.
                Algorithm-key will be ignored
                Example:
                    {
                        'power_station': {
                            'max_power': float,
                            'duration_time_step': float,
                            'positions': int
                        },
                        'vehicles': [
                            {
                                'position': int,
                                'capacity_max': flaot,
                                'charging_power_min': float,
                                'charging_power_max': float,
                                'power_dissipation' : float,
                                'capacity_history': list[float],
                                'limit': float,
                                'time_steps': int,
                                'exp_charge': float
                            }
                        ]
                    }

        Returns:
            Environment: Environment created from the data.
        """
        # Create vehicles
        vehicles = []
        for vehicle in data['vehicles']:
            vehicles.append(Vehicle.generate_vehicle_from_json(vehicle))

        # Create env
        power_station = data['power_station']
        env = cls(max_power=power_station['max_power'],
                positions=power_station['positions'],
                duration_time_step=power_station['duration_time_step'],
                is_train = False
                )

        env.vehicles = vehicles

        return env

    def to_json(self) -> dict:
        """Generate JSON for given Environment

        Returns:
            dict: Current Environment-state as dictionary. Example:
                {
                    'power_station': {
                        'max_power': float,
                        'duration_time_step': float,
                        'positions': int
                    },
                    'vehicles': [
                        {
                            'position': int,
                            'capacity_max': flaot,
                            'charging_power_min': float,
                            'charging_power_max': float,
                            'power_dissipation' : float,
                            'capacity_history': list[float],
                            'limit': float,
                            'time_steps': int,
                            'exp_charge': float
                        }
                    ]
                }
        """
        data = {
            'power_station': {
                'max_power': self.max_power,
                'duration_time_step': self.duration_time_step,
                'positions': self.positions,
            }
        }
        data['vehicles'] = [vehicle.__dict__ for vehicle in self.vehicles]
        return data
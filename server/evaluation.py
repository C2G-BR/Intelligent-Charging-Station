from __future__ import annotations
from typing import Union
from os.path import join
from json import dump

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import io

from model.agent import DDPGAgent, FCFSAgent
from model.env import Environment
from model.recorder import Recorder

def visualize_environments(
    state:dict,
    reward:dict,
    steps:list,
    current_step:int,
    write_to_buffer:bool=True
) -> Union[str, None]:
    """Display the current charging state in a bar chart

    Args:
        state (dict): Current capacity on all positions
        reward (dict): Total reward for the last 10 steps
        steps (list): List containing the last 10 steps
        current_step (int): index of the current episode
        write_to_buffer (bool, optional): _description_. Defaults to True.

    Returns:
        Union[str, None]: Returns if write_to_buffer is True a within the
        buffer saved figured as string; otherwise None
    """

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(1, 2, num=1)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.rcParams.update({'font.size': 28})
    fig.suptitle(f'Model comparison - step {current_step}')
    plt.subplots_adjust(top = 0.9, bottom = 0.1, wspace = 0.25)
    ax1 = axes[0]
    ax2 = axes[1]

    total_width = 0.8
    single_width=1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(state)
    bar_width = total_width / n_bars
    bars = []

    for i, (name, values) in enumerate(state.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
    
        for x, y in enumerate(values):
            bar = ax1.bar(x + x_offset, y, width=bar_width * single_width, 
                color=colors[i % len(colors)])
        
        bars.append(bar[0])
    ax1.legend(bars, state.keys(), loc='upper right')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Current Capacity')
    ax1.set_title('Comparison current state')

    for i, (name, values) in enumerate(reward.items()):
        # values = [sum(values[:i]) for i in range(len(values))]
        ax2.plot(steps, values, color=colors[i % len(colors)], label=name)

    ax2.legend(loc='upper right')
    ax2.set_xlabel(f'Last {len(steps)} steps')
    ax2.set_ylabel('Reward')
    ax2.set_title('Comparison reward')
    plt.pause(1)

    if write_to_buffer:
        with io.BytesIO() as output:
            plt.savefig(output, format='png')
            return output.getvalue()
        
    return None

def get_env_capacity(env: Environment) -> list:
    """Calculates capacity in percentage for each position within Environment

    Args:
        env (Environment): Current environment

    Returns:
        list: capacity in percentage for every position
    """
    heights = []
    available_positions = env.get_available_positions()
    # Retrive the current values for all positions.
    for p in range(env.positions):
        if p in available_positions:
            heights.append(0)
        else:
            vehicle = env.get_vehicle_by_position(position=p)
            heights.append(vehicle.capacity_in_percentage()*100)
    return heights

if __name__ == '__main__':

    """Evalutaing both agents against each other
    """

    RECORD = True      # should the comparison be recorded
    NUM_STEPS = 50     # number of epochs to run the simulation
    MAX_POWER = 49     # max power [kw] for charging station
    POSITIONS = 20     # number of slots/positions within charging station
    TIME_INTERVAL = 0.2
    SHOULD_DUMP = True

    if RECORD:
        recorder = Recorder(fps = 2, height=1080, width=1920, folder = 'output')
        recorder.init_new_video(f'video1')

    # create environment
    # Specify the models that should be compared
    agent = DDPGAgent(n_actions=POSITIONS, input_dims=[120])
    agent.load_agent(join('model/trained_model'))

    models = {
        'fcfs': {
            'agent': FCFSAgent(),
            'normalize': False,
            'environment':
                Environment(
                    max_power=MAX_POWER,
                    positions=POSITIONS,
                    duration_time_step=TIME_INTERVAL
                ),
            'observation': [],
            'reward': [],
            'done': [],
            'heights': []
        },
        'ddpg': {
            'agent': agent,
            'normalize': True,
            'environment':
                Environment(
                    max_power=MAX_POWER, 
                    positions=POSITIONS,
                    duration_time_step=TIME_INTERVAL
                ),
            'observation': [],
            'reward': [],
            'done': [],
            'heights': []
        }
    }

    for model, values in models.items():
        env = values['environment']
        agent = values['agent']
        np.random.seed(0)

        for i in range(NUM_STEPS):

            if i == 0:
                observation, reward, done = env.reset()
                models[model]['observation'].append(observation)
                models[model]['reward'].append(reward)
                models[model]['done'].append(done)
                models[model]['heights'].append(get_env_capacity(env))


            observation = models[model]['observation'][-1]
            if values['normalize']:
                actions = agent.predict(observation=observation)
                actions = DDPGAgent.normalize_actions(actions, env.max_power)
            else:
                actions = agent.predict(
                    observation=observation, 
                    max_power=env.max_power
                )

            observation, reward, done = env.step(
                charging_power_per_vehicle=actions
            )

            models[model]['observation'].append(observation)
            models[model]['reward'].append(reward)
            models[model]['done'].append(done)
            models[model]['heights'].append(get_env_capacity(env))

    if SHOULD_DUMP:
        for model in ['fcfs', 'ddpg']:
            for key in ['agent', 'environment', 'normalize']:
                models[model].pop(key, None)
        with open('output/small_kw.json', 'w') as f:
            dump(models, f)

    for i in range(NUM_STEPS):
        state = {}
        reward = {}
        for model, values in models.items():
            state[model] = values['heights'][i]
            start = i - 10 if i > 10 else 0
            
            r = values['reward']
            ranges = range(0, i)
            r = [sum(r[:i]) for i in ranges]
            reward[model] = r
            steps = list(ranges)
            
        frame = visualize_environments(state, reward, steps, i, RECORD)
        if RECORD:
            recorder.add_image(frame)

    if RECORD:
        recorder.close_recording()
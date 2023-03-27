import argparse
import logging
import numpy as np
import os
import random
import torch as T

from collections import deque
from itertools import count
from os.path import join
from torch.utils.tensorboard import SummaryWriter

from model.agent import DDPGAgent
from model.env import Environment
from model.noise import OrnsteinUhlenbeckActionNoise

def get_logger(path:str, debug:bool = False) -> logging.Logger:
    """Retrieve logger

    The returned logger writes to a file and streams the logs to the console.

    Args:
        path (str): Location for the log file.
        debug (bool, optional): Whether to use debug mode for logging. Defaults
            to False.

    Returns:
        logging.Logger: Logger that streams and writes logs.
    """
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('ICS')
    logger.setLevel(level=level)
    fh = logging.FileHandler(join(path, 'logs.log'), mode='a')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def train(args:argparse.Namespace) -> None:
    """Train a Deep Deterministic Policy Graient Agent

    Args:
        args (argparse.Namespace): Arguments for the training.

    Raises:
        Exception: Id already exists and training is not choosen to continue
            with the given agent.
    """
    experiment_id = args.id
    seed = args.seed

    # Create folder structure.
    experiments_path = join('experiments')
    experiment_path = join(experiments_path, experiment_id)
    tensorboard_path = join(experiments_path, 'tensorboard')
    save_path = join(experiment_path, 'saves')

    if not os.path.exists(experiments_path):
        os.mkdir(experiments_path)

    if os.path.exists(experiment_path):
        if not args.load_agent:
            raise Exception(f'Experiment with id {experiment_id} already '
                'exists. Choose a different one.')
    else:
        os.mkdir(experiment_path)
        os.mkdir(save_path)


    # Initialize tracking utilities.
    writer = SummaryWriter(join(tensorboard_path, experiment_id))
    logger = get_logger(path=experiment_path, debug=args.debug)
    logger.info(args)

    # Settings
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    INPUT_DIM_PER_POSITION = 6

    # Get objects.
    env = Environment(max_power=args.max_power, max_steps=args.max_steps,
        positions=args.positions, duration_time_step=args.duration_time_step,
        is_train=True, seed=seed)
    input_dims = [env.positions * INPUT_DIM_PER_POSITION]
    noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.positions),
        sigma=args.std_noise)
    agent = DDPGAgent(pi_lr=args.actor_lr, q_lr=args.critic_lr,
        input_dims=input_dims, batch_size=args.batch_size, noise=noise,
        gamma=args.gamma, min_replay_size=args.min_replay_size,
        replay_buffer_size=args.max_replay_size, tau=args.tau,
        n_actions=env.positions, init_epsilon=args.epsilon_init,
        epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min)
    reward_history = deque(maxlen=args.reward_rolling_window_size)

    # Load agent if path is given and files exist.
    start_episode = 0
    if args.load_agent:
        logger.info('Trying to load agent.')
        agent.load_agent(save_path=join(save_path, str(args.load_episode)))
        logger.info('Agent completely loaded.')
        start_episode = args.load_episode + 1

    for episode in range(start_episode, start_episode+args.num_episodes):
        # Training
        obs, _, _ = env.reset()
        noise.reset()
        acc_reward = 0.0
        actor_loss = []
        critic_loss = []

        for step in count():
            action = agent.predict_eps(observation=obs)
            input_action = DDPGAgent.normalize_actions(action, env.max_power)
            
            # Take step in environment
            new_obs, reward, done = env.step(input_action)
            acc_reward += reward

            # Store experience
            agent.add_experience(obs, action, reward, new_obs, done)
            obs = new_obs

            # Train agent
            if step % args.update_steps == 0:
                if agent.replay_buffer.size() > agent.min_replay_size:
                    # TODO: Return tuple of loss?
                    loss = agent.train()
                    actor_loss.append(loss['actor_loss'].cpu())
                    critic_loss.append(loss['critic_loss'].cpu())
                    agent.update()

            if done:
                break

        agent.update_eps()

        # Tracking stats.        
        reward_history.append(acc_reward)

        mean_actor_loss = np.mean(actor_loss) if len(actor_loss) > 0 else 0
        mean_critic_loss = np.mean(critic_loss) if len(critic_loss) > 0 else 0
        writer.add_scalar('acc_reward', acc_reward, global_step=episode)
        writer.add_scalar('steps', step, global_step=episode)
        writer.add_scalar('avg_actor_loss', mean_actor_loss,
            global_step=episode)
        writer.add_scalar('avg_critic_loss', mean_critic_loss,
            global_step=episode)
        logger.info(f'Episode: {episode:6d} | Episode reward: '
            f'{acc_reward:7.2f} | Average reward: '
            f'{np.mean(reward_history):7.2f}')

        # Evaluation
        if episode % args.evaluation_episodes == 0:
            evaluation_rewards = []
            for _ in range(args.num_evaluation_episodes):
                obs, _, _ = env.reset()
                rewards = 0

                while True:
                    action = agent.predict(obs)
                    input_action = DDPGAgent.normalize_actions(action,
                        env.max_power)

                    new_obs, reward, done = env.step(input_action)

                    obs = new_obs
                    rewards += reward

                    if done:
                        break

                evaluation_rewards.append(rewards)

            agent.save_agent(os.path.join(save_path, str(episode)))
            writer.add_scalar('acc_eval_reward', np.mean(evaluation_rewards),
                global_step=episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train Deep Deterministic Policy '
        'Gradient', description='Training of a DDPG Agent.')

    # Training
    parser.add_argument('-id', '--identifier', type=str, dest='id',
        required=True, help='Unique identifier used to identify the run.')
    parser.add_argument('-s', '--seed', type=int, default=0, dest='seed',
        help='Seed used for random generators.')
    parser.add_argument('-us', '--update-steps', type=int, default=1,
        dest='update_steps', help='Number of steps until an update is '
        'performed.')
    parser.add_argument('-ne', '--num-episodes', type=int, default=int(2e4),
        dest='num_episodes', help='The number of episodes to play.')
    parser.add_argument('-evep', '--evaluation-episodes', type=int, default=10,
        dest='evaluation_episodes', help='Number of episodes until an '
        'evaluation is performed.')
    parser.add_argument('-nevep', '--num-evaluation-episodes', type=int,
        default=10, dest='num_evaluation_episodes', help='Number of episodes '
        'performed within an evaluation.')
    parser.add_argument('-ws', '--reward-rolling-window-size', type=int,
        default=100, dest='reward_rolling_window_size', help='Size of rolling '
        'window used for logging the reward.')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug',
        help='Whether to use debug mode for logging.')
    
    # Agent
    parser.add_argument('-alr', '--actor-learning-rate', type=float,
        default=0.001, dest='actor_lr', help='Learning rate for the actor.')
    parser.add_argument('-clr', '--critic-learning-rate', type=float,
        default=0.002, dest='critic_lr', help='Learning rate for the critic.')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
        dest='batch_size', help='The number of experiences choosen to update '
        'the train model.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, dest='gamma',
        help='The discount rate for future expected rewards.')
    parser.add_argument('-mirs', '--min-replay-size', type=int, default=256,
        dest='min_replay_size', help='The minimum number of experiences kept.')
    parser.add_argument('-mars', '--max-replay-size', type=int,
        default=int(1e5), dest='max_replay_size', help='The maximum number of '
        'experiences kept.')
    parser.add_argument('-t', '--tau', type=float, default=0.005, dest='tau',
        help='Value for the soft update.')
    parser.add_argument('-la', '--load-agent', action='store_true',
        dest='load_agent', help='Whether to load parameters of an existing '
        'agent.')
    parser.add_argument('-le', '--load-episode', type=int, dest='load_episode',
        required=False, help='The episode of the existing agent used to load. '
        'This only applies if load-agent is True.')
    parser.add_argument('-ei', '--epsilon-init', type=float,
        dest='epsilon_init', default=0.99, help='The initial value for epsilon '
        'as in the decaying epsilon greedy scheme.')
    parser.add_argument('-ed', '--epsilon-decay', type=float,
        dest='epsilon_decay', default=0.997, help='The decay value for epsilon '
        'as in the decaying epsilon greedy scheme.')
    parser.add_argument('-em', '--epsilon-min', type=float, dest='epsilon_min',
        default=0.05, help='The minimum value for epsilon as in the decaying '
        'epsilon greedy scheme.')

    # Noise
    parser.add_argument('-std', '--std-noise', type=float, default=0.2,
        dest='std_noise', help='Scale of the noise in form of standard '
        'deviation.')
    
    # Environment
    parser.add_argument('-mp', '--max-power', type=float, default=150,
        dest='max_power', help='Available power in kWh that can be distributed '
        'by the agent.')
    parser.add_argument('-p', '--positions', type=int, default=8,
        dest='positions', help='Number of parking lots.')
    parser.add_argument('-ms', '--max-steps', type=int, default=200,
        dest='max_steps', help='Maximum number of steps each episode has.')
    parser.add_argument('-dts', '--duration-time-step', type=float, default=1,
        dest='duration_time_step', help='The time difference between each time '
        'step in h.')

    args = parser.parse_args()
    train(args)
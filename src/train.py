#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
from osim.env import RunEnv
from multiprocessing import Pool
import multiprocessing
import urllib
import http.client
import json, pickle
import time

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym():
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    # env = gym.make(env_name)
    env = RunEnv(visualize=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim

def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    pass
#     print("Episode run")
#     obs = env.reset(difficulty=2)
#     observes, actions, rewards, unscaled_obs = [], [], [], []
#     done = False
#     step = 0.0
#     scale, offset = scaler.get()
#     scale[-1] = 1.0  # don't scale time step feature
#     offset[-1] = 0.0  # don't offset time step feature
#     while not done:
#         if animate:
#             env.render()
#         obs = np.asarray(obs)
#         obs = obs.astype(np.float64).reshape((1, -1))
#         obs = np.append(obs, [[step]], axis=1)  # add time step feature
#         unscaled_obs.append(obs)
#         obs = (obs - offset) * scale  # center and scale observations
#         observes.append(obs)
#         action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
#         actions.append(action)
#         obs, reward, done, _ = env.step(action[0])
#         if not isinstance(reward, float):
#             reward = np.asscalar(reward)
#         rewards.append(reward)
#         step += 1e-3  # increment time step feature
#
#     trajectory = {'observes': np.concatenate(observes),
#                  'actions': np.concatenate(actions),
#                  'rewards': np.array(rewards, dtype=np.float64),
#                  'unscaled_obs': np.concatenate(unscaled_obs)}
#     return trajectory
#     # return (np.concatenate(observes), np.concatenate(actions),
#     #         np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes, cores, port):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    if cores > 1:
        conn_str = '127.0.0.1:' + str(port)
        conn = http.client.HTTPConnection(conn_str)
        headers = {
                "cache-control": "no-cache"
                }
        query = {
                "chk_dir": logger.log_dir,
                "episodes": episodes,
                "cores": cores
                }
        encoded_query = urllib.parse.urlencode(query)
        response = None
        while response != 'OK':
            conn.request("GET", "/get_episodes?" + encoded_query,
                    headers=headers)
            res = conn.getresponse()
            response = json.loads(res.read())['Success']
            trajectories = pickle.load(open(logger.log_dir + '/episodes_latest', 'rb'))
    else:
        trajectories = [run_episode(env, policy, scaler) for x in range(episodes)]
    total_steps = sum([t['observes'].shape[0] for t in trajectories])
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'MaxReward': np.max([t['rewards'].sum() for t in trajectories]),
                'MinReward': np.min([t['rewards'].sum() for t in trajectories]),
                'MeanLen': np.mean([len(t['rewards']) for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, snapshot,
        cores, port):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym()
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)
    aigym_path = os.path.join('/tmp', env_name, now)
    # env = wrappers.Monitor(env, aigym_path, force=True)
    val_func = NNValueFunction(obs_dim, logger, snapshot=snapshot)
    policy = Policy(obs_dim, act_dim, kl_targ, logger, snapshot=snapshot)
    if snapshot != None:
        print("loading scaler from", snapshot)
        scaler = pickle.load(open(snapshot + '/scaler_latest', 'rb'))
    else:
        scaler = Scaler(obs_dim)
        pickle.dump(scaler, open(logger.log_dir + '/scaler_latest', 'wb'))
        pickle.dump(scaler, open(logger.log_dir + '/scaler_0', 'wb'))
        # run a few episodes of untrained policy to initialize scaler:
        run_policy(env, policy, scaler, logger, episodes=5, cores=cores, port=port)
        print(scaler.means)
        print(scaler.vars)
    episode = 0
    pickle.dump(scaler, open(logger.log_dir + '/scaler_latest', 'wb'))
    pickle.dump(scaler, open(logger.log_dir + '/scaler_0' + str(episode), 'wb'))
    while episode < num_episodes:
        begin = time.time()
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size, cores=cores, port=port)
        episode += len(trajectories)
        pickle.dump(scaler, open(logger.log_dir + '/scaler_latest', 'wb'))
        pickle.dump(scaler, open(logger.log_dir + '/scaler_' + str(episode), 'wb'))
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger, episode)  # update policy
        logger.log({'TimeTaken': time.time() - begin})
        val_func.fit(observes, disc_sum_rew, logger, episode)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-c', '--cores', type=int,
                        help='Number of parallel threads',
                        default=2)
    parser.add_argument('-s', '--snapshot', type=str,
                        help='Snapshot folder')

    parser.add_argument('-p', '--port', type=int,
                        help='Parallel worker port',
                        default=8018)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

import argparse, pickle, os
from osim.http.client import Client
from osim.env import RunEnv
from tqdm import tqdm
import logging
from parallel_episode_server import get_action_from_obs
import numpy as np

logger = logging.getLogger('osim.http.client')
logger.setLevel(logging.CRITICAL)

remote_base = 'http://grader.crowdai.org:1729'
token = 'b97ecc86c6e23bda7b2ee8771942cb9c'

def rollout_episode(chk_dir, submit):
    scaler_file = chk_dir + '/scaler_latest'
    scaler = pickle.load(open(scaler_file, 'rb'))
    import tensorflow as tf
    sess = tf.Session()
    latest_chkp_file = tf.train.latest_checkpoint(chk_dir, latest_filename='policy_checkpoint')
    meta_file = latest_chkp_file + '.meta'
    if not os.path.exists(meta_file):
        meta_file = chk_dir + '/policy-model-0.meta'
    meta_graph = tf.train.import_meta_graph(meta_file)
    meta_graph.restore(sess, latest_chkp_file)
    obs_ph = tf.get_collection('obs_ph_chk')[0]
    sampled_act = tf.get_collection('sampled_act_chk')[0]
    if submit == 1:
        client = Client(remote_base)
        obs = client.env_create(token)
    else:
        client = RunEnv(visualize=False)
        obs = client.reset(difficulty=2)
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    i = 0
    j = 0
    with tqdm(total=3000) as step_bar:
        while True:
            i += 1
            step_bar.update(1)
            obs = np.asarray(obs)
            obs = obs.astype(np.float64).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale
            observes.append(obs)
            action = get_action_from_obs(sess, obs_ph, sampled_act, obs)
            actions.append(action)
            if submit == 1:
                obs, reward, done, _ = client.env_step(action[0].tolist(), True)
            else:
                # obs, reward, done, _ = client.step(np.clip(action[0], 0, 1))
                obs, reward, done, _ = client.step(action[0])
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            step += 1e-3
            if done:
                j += 1
                i = 0
                print(j, "episode done")
                print(i, "took these many steps")
                print(np.sum(rewards), 'total reward')
                step = 0.0
                if submit == 1:
                    obs = client.env_reset()
                else:
                    obs = client.reset(difficulty=2)
                    if j == 3:
                        break
                if not obs:
                    break
    trajectory = {'observes': np.concatenate(observes),
                  'actions': np.concatenate(actions),
                  'rewards': np.array(rewards, dtype=np.float64),
                  'unscaled_obs': np.concatenate(unscaled_obs)}
    print("Reward is", np.sum(rewards))
    if submit == 1:
        client.submit()
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=str)
    parser.add_argument("-s", "--submit", type=int, default=0, help="1 to submit to server")
    parser.add_argument("-v", "--visualize", type=int, default=0, help="1 to visualize")
    args = parser.parse_args()
    rollout_episode(args.snapshot, args.submit)

if __name__ == '__main__':
    main()

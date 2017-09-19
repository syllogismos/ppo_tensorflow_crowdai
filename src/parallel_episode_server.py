from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import urllib.parse as urlparse

from osim.env import RunEnv
import numpy as np
from utils import Scaler
import multiprocessing
import pickle, random
import copy
from utils import build_features


PORT_NUMBER = 8018


def dump_episodes(chk_dir, episodes, cores, filter_type):
    scaler_file = chk_dir + '/scaler_latest'
    scaler = pickle.load(open(scaler_file, 'rb'))
    seeds = [random.randint(0, 100000000) for x in range(episodes)]
    print("Current Iteration seeds are")
    print(seeds)
    print("=========================")
    p = multiprocessing.Pool(cores, maxtasksperchild=1)
    picklable_argument = zip(
                             [scaler]*episodes,
                             [chk_dir]*episodes,
                             [filter_type]*episodes,
                             [random.randint(0, 100000000) for x in range(episodes)]
                            )
    tras = p.map(run_episode_from_last_checkpoint,
                 picklable_argument)
    p.close()
    p.join()
    print("Current Iteration seeds are")
    print(seeds)
    print("=========================")
    episodes_file = chk_dir + '/episodes_latest'
    pickle.dump(tras, open(episodes_file, 'wb'))

def get_random(x):
    return random.randint(0, 100000000)
def get_random_np(x):
    return np.random.randint(0, 100000000)
def get_sample_seed(episodes):
    # seeds = [random.randint(0, 10000000) for x in range(episodes)]
    # print(seeds)
    # seeds_np = [np.random.randint(0, 100000000) for x in range(episodes)]
    # print(seeds_np)
    p = multiprocessing.Pool(episodes)
    seeds = p.map(get_random, range(2))
    seeds_np = p.map(get_random_np, range(2))
    p.close()
    p.join()
    print(seeds)
    print(seeds_np)


def run_episode_from_last_checkpoint(pickled_object):
    """
    Load the last checkpoint from the current folder, and using that
    checkpoint run episodes parallely to collect the episodes

    Args:
    pickled_object = (scaler, chk_dir, filter_type, seed)
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        chk_dir: the logger object
        filter_type: type of the filter to apply to obs
        seed: seed to be passed to runenv

    Returns: 4-typle of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    import tensorflow as tf
    scaler = pickled_object[0]
    chkp_dir = pickled_object[1]
    filter_type = pickled_object[2]
    seed = pickled_object[3]
    sess = tf.Session()
    latest_chkp_file = tf.train.latest_checkpoint(chkp_dir, latest_filename='policy_checkpoint')
    meta_graph = tf.train.import_meta_graph(latest_chkp_file + '.meta')
    print(latest_chkp_file)
    meta_graph.restore(sess, latest_chkp_file)
    obs_ph = tf.get_collection('obs_ph_chk')[0]
    sampled_act = tf.get_collection('sampled_act_chk')[0]
    env = RunEnv(visualize=False)
    cur_obs = env.reset(difficulty=2, seed=seed)
    old_obs = copy.copy(cur_obs)
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    while not done:
        obs = build_features(cur_obs, old_obs, filter_type)
        obs = np.asarray(obs)
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale
        observes.append(obs)
        action = get_action_from_obs(sess, obs_ph, sampled_act, obs)
        actions.append(action)
        old_obs = copy.copy(cur_obs)
        cur_obs, reward, done, _ = env.step(action[0])
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3
    trajectory = {'observes': np.concatenate(observes),
                 'actions': np.concatenate(actions),
                 'rewards': np.array(rewards, dtype=np.float64),
                 'unscaled_obs': np.concatenate(unscaled_obs)}
    return trajectory

def get_action_from_obs(sess, obs_ph, sampled_act, obs):
    feed_dict = {obs_ph: obs}
    return np.clip(sess.run(sampled_act, feed_dict=feed_dict).reshape((1, -1)).astype(np.float64),
                   0.0,
                   1.0)


class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if '/ping' in self.path:
            print(self.path)
            parsed_url = urlparse.urlparse(self.path)
            print(urlparse.parse_qs(parsed_url.query))
            print('lmao it worked')
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'anil': 'tanu'}), 'utf8'))
            return
        if '/get_episodes' in self.path:
            parsed_url = urlparse.urlparse(self.path)
            query = urlparse.parse_qs(parsed_url.query)
            episodes = int(query['episodes'][0])
            chk_dir = query['chk_dir'][0]
            cores = int(query['cores'][0])
            filter_type = str(query['filter_type'][0])
            print(chk_dir)
            print(episodes)
            print(filter_type)
            dump_episodes(chk_dir, episodes, cores, filter_type)
            # s = Scaler(42)
            # traj = mp_test(s)
            # pickle.dump(traj, open('traj.pkl', 'wb'))
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'Success': 'OK'}), 'utf8'))

        if '/seed_test' in self.path:
            get_sample_seed(2)
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'Success': 'OK'}), 'utf8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=PORT_NUMBER)
    args = parser.parse_args()
    server = HTTPServer((args.listen, args.port), myHandler)
    print('Server started on', args)
    server.serve_forever()

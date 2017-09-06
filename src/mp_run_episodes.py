from osim.env import RunEnv
import numpy as np


import multiprocessing

def mp_test(s):
    p = multiprocessing.Pool(2)
    tras = p.map(run_episode_from_last_checkpoint, [(s, 'a')]*4)
    p.close()
    p.join()
    return tras


def run_episode_from_last_checkpoint(pickled_object):
    """
    Load the last checkpoint from the current folder, and using that
    checkpoint run episodes parallely to collect the episodes

    Args:
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: the logger object

    Returns: 4-typle of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    # observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
    # total_steps += observes.shape[0]
    # trajectory = {'observes': observes,
    #               'actions': actions,
    #               'rewards': rewards,
    #               'unscaled_obs': unscaled_obs}
    import tensorflow as tf
    scaler = pickled_object[0]
    chkp_dir = pickled_object[1]
    print(chkp_dir)
    # g = tf.Graph()
    # print(g)
    sess = tf.Session() #graph = g)
    print(sess)
    # with g.as_default():
    print('in with')
    chkp_dir = '/home/ubuntu/pat-cody/log-files/RunEnv_test2/Sep-02_11:57:45'
    meta_graph = tf.train.import_meta_graph(chkp_dir + '/policy-model-0.meta')
    print('meta graph')
    print(tf.train.latest_checkpoint(chkp_dir, latest_filename='policy_checkpoint'))
    meta_graph.restore(sess, tf.train.latest_checkpoint(chkp_dir, latest_filename='policy_checkpoint'))
    print('restroring meta graph')
    obs_ph = tf.get_collection('obs_ph_chk')[0]
    print('obs_ph')
    sampled_act = tf.get_collection('sampled_act_chk')[0]
    print('sampled act')
    print('out of with')
    env = RunEnv(visualize=False)
    print(env)
    obs = env.reset(difficulty=2)
    print(obs)
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    while not done:
        obs = np.asarray(obs)
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale
        observes.append(obs)
        action = get_action_from_obs(sess, obs_ph, sampled_act, obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action[0])
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3
    trajectory = {'observes': np.concatenate(observes),
                 'actions': np.concatenate(actions),
                 'rewards': np.array(rewards, dtype=np.float64),
                 'unscaled_obs': np.concatenate(unscaled_obs)}
    return trajectory
    # return (np.concatenate(observes), np.concatenate(actions),
    #         np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

def get_action_from_obs(sess, obs_ph, sampled_act, obs):
    print('getting action from obs')
    feed_dict = {obs_ph: obs}
    return sess.run(sampled_act, feed_dict=feed_dict).reshape((1, -1)).astype(np.float64)



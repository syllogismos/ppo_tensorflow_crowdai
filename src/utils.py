"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os
import shutil
import glob
import csv
import copy


def build_features(obs, old_obs, filter_type):
    if filter_type == 'IDENTITY':
        return copy.copy(obs)
    elif filter_type == 'FILTER1':
        return get_computed_observation_filter1(obs, old_obs)

def get_computed_observation_filter1(obs, old_obs):
    """
    given previous observation and current observation
    add more features, like previous
    """
    new_obs = copy.copy(obs)
    pelvis = obs[1]
    head = obs[22]
    com = obs[18]
    new_obs.append(pelvis - head)
    new_obs.append(pelvis - com)
    new_obs.append(head - com)
    new_obs.append(pelvis - old_obs[22])
    new_obs.append(pelvis - old_obs[24])
    new_obs.append(pelvis - old_obs[26])
    new_obs.append(pelvis - old_obs[28])
    new_obs.append(pelvis - old_obs[30])
    new_obs.append(pelvis - old_obs[32])
    new_obs.append(pelvis - old_obs[34])
    new_obs.append(head - old_obs[22])
    new_obs.append(head - old_obs[24])
    new_obs.append(head - old_obs[26])
    new_obs.append(head - old_obs[28])
    new_obs.append(head - old_obs[30])
    new_obs.append(head - old_obs[32])
    new_obs.append(head - old_obs[34])
    new_obs.append(obs[22] - old_obs[22])
    new_obs.append(obs[23] - old_obs[23])
    new_obs.append(obs[24] - old_obs[24])
    new_obs.append(obs[25] - old_obs[25])
    new_obs.append(obs[26] - old_obs[26])
    new_obs.append(obs[27] - old_obs[27])
    new_obs.append(obs[28] - old_obs[28])
    new_obs.append(obs[29] - old_obs[29])
    new_obs.append(obs[30] - old_obs[30])
    new_obs.append(obs[31] - old_obs[31])
    new_obs.append(obs[32] - old_obs[32])
    new_obs.append(obs[33] - old_obs[33])
    new_obs.append(obs[34] - old_obs[34])
    new_obs.append(obs[35] - old_obs[35])
    return new_obs

class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        path = os.path.join('log-files', logname, now)
        path = os.path.abspath(path)
        os.makedirs(path)
        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)
        self.log_dir = path
        path = os.path.join(path, 'log.csv')
        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def get_file_name(self, suffix):
        return os.path.join(self.log_dir, suffix)

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.3f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()

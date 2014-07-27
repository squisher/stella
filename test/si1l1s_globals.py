import time
from math import log

import numpy as np

from test import *  # noqa
import mtpy
import stella

EXPSTART = 0.2


def prepare(params):
    global K, rununtiltime, koffp, kcat, delta, leg, substrate, obs_i, observations
    K = params['K']
    rununtiltime = params['rununtiltime']
    mtpy.mt_seed32new(params['seed'])
    koffp = params['koffp']
    kcat = params['r']

    delta = (log(rununtiltime) - log(EXPSTART)) / float(K - 1)
    leg = 0
    substrate = 0
    obs_i = 0
    observations = np.zeros(shape=K, dtype=int)


def uniform():
    return mtpy.mt_drand()


def exp(p):
    u = 1.0 - uniform()
    return -log(u) / p


def makeObservation():
    """Called from run()"""
    global observations, leg, obs_i, next_obs_time
    observations[obs_i] = leg
    obs_i += 1

    next_obs_time = getNextObsTime()


def getNextObsTime():
    """Called from run()"""
    global obs_i, EXPSTART, rununtiltime, delta
    if obs_i == 0:
        return EXPSTART
    if obs_i == K - 1:
        return rununtiltime

    return exp(log(EXPSTART) + delta * obs_i)


def step():
    """Called from run()"""
    global leg, substrate
    if leg == 0:
        leg += 1
    else:
        u1 = uniform()
        if u1 < 0.5:
            leg -= 1
        else:
            leg += 1
    if leg == substrate:
        substrate += 1


def isNextObservation():
    global t, next_obs_time, obs_i, K
    return t > next_obs_time and obs_i < K


def run():
    global t, next_obs_time, obs_i, K, rununtiltime, leg, substrate
    t = 0.0
    next_obs_time = getNextObsTime()

    # TODO: Declaring R here is not necessary in Python! But llvm needs a
    # it because otherwise the definition of R does not dominate the use below.
    R = 0.0
    while obs_i < K and t < rununtiltime:
        if leg < substrate:
            R = koffp
        else:
            R = kcat
        t += exp(R)

        while isNextObservation():
            makeObservation()

            step()


class BaseSettings(object):

    def setDefaults(self):
        self.settings = {
            'seed': [int(time.time()), int],
            'r': [0.1, float],
            'koff': [1.0, float],
            'radius': [10, int],
            'nlegs': [2, int],
            'gait': [2, int],
            'dim': [2, int],
            'nspiders': [1, int],     # not completely functional yet
            'elapsedTime': [self.elapsedTime, lambda x:x],
        }

    def elapsedTime(self):
        return time.time() - self.start_time

    def __init__(self, argv=[]):
        self.start_time = time.time()

        self.setDefaults()

        # parse command line arguments to overwrite the defaults
        for key, _, val in [s.partition('=') for s in argv]:
            self[key] = val

    def __setitem__(self, k, v):
        if k in self.settings:
            self.settings[k][0] = self.settings[k][1](v)
        else:
            self.settings[k] = [v, type(v)]

    def __getitem__(self, k):
        return self.settings[k][0]

    def __str__(self):
        r = '{'
        for k, (v, type_) in self.settings.items():
            if isinstance(type_,  FunctionType):
                continue
            r += str(k) + ':' + str(v) + ', '
        return r[:-2] + '}'


class Settings(BaseSettings):

    def setDefaults(self):
        self.settings = {
            'seed': [int(time.time()), int],
            'r': [0.1, float],
            'koffp': [1.0, float],
            'K': [10, int],
            'rununtiltime': [1e3, float],
            'elapsedTime': [self.elapsedTime, lambda x:x],
        }


def prototype(params):
    s = Settings(params)

    prepare(s)
    run()
    py = np.array(observations)  # save the global result variable

    prepare(s)
    stella.wrap(run)()
    assert id(py) != id(observations)
    st = observations

    assert all(py == st)


@mark.parametrize('args', [['seed=42'], ['seed=63'], ['seed=123456'],
                           ['rununtiltime=1e4', 'seed=494727']])
def test1(args):
    prototype(args)

timed = timeit(prototype)


def bench1():
    timed(['seed=42', 'rununtiltime=1e8'])

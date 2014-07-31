import time
from math import log

import numpy as np

from test import *  # noqa
import mtpy
import stella

EXPSTART = 0.2
class Spider(object):
    def __init__(self, params, observations):
        self.K = params['K']
        self.rununtiltime = params['rununtiltime']
        mtpy.mt_seed32new(params['seed'])
        self.koffp = params['koffp']
        self.kcat = params['r']

        self.delta = (log(self.rununtiltime) - log(EXPSTART)) / float(self.K - 1)
        self.leg = 0
        self.substrate = 0
        self.obs_i = 0
        self.observations = observations
        # LANG: Init below required before entering stella!
        # TODO: Static analysis could discover the use in the original location
        self.t = 0.0
        self.next_obs_time = 0.0


def uniform():
    return mtpy.mt_drand()


def exp(p):
    u = 1.0 - uniform()
    return -log(u) / p


def makeObservation(sp):
    """Called from run()"""
    sp.observations[sp.obs_i] = sp.leg
    sp.obs_i += 1

    sp.next_obs_time = getNextObsTime(sp)


def getNextObsTime(sp):
    """Called from run()"""
    if sp.obs_i == 0:
        return EXPSTART
    if sp.obs_i == sp.K - 1:
        return sp.rununtiltime

    return exp(log(EXPSTART) + sp.delta * sp.obs_i)


def step(sp):
    """Called from run()"""
    if sp.leg == 0:
        sp.leg += 1
    else:
        u1 = uniform()
        if u1 < 0.5:
            sp.leg -= 1
        else:
            sp.leg += 1
    if sp.leg == sp.substrate:
        sp.substrate += 1


def isNextObservation(sp):
    return sp.t > sp.next_obs_time and sp.obs_i < sp.K


def run(sp):
    # LANG: Init below moved to Spider.__init__
    #sp.t = 0.0
    sp.next_obs_time = getNextObsTime(sp)

    # TODO: Declaring R here is not necessary in Python! But llvm needs a
    # it because otherwise the definition of R does not dominate the use below.
    R = 0.0
    while sp.obs_i < sp.K and sp.t < sp.rununtiltime:
        if sp.leg < sp.substrate:
            R = sp.koffp
        else:
            R = sp.kcat
        sp.t += exp(R)

        while isNextObservation(sp):
            makeObservation(sp)

            step(sp)


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

    py = np.zeros(shape=s['K'], dtype=int)
    sp = Spider(s, py)
    run(sp)

    st = np.zeros(shape=s['K'], dtype=int)
    sp = Spider(s, st)
    stella.wrap(run)(sp)

    assert id(py) != id(observations)
    assert all(py == st)


@mark.parametrize('args', [['seed=42'], ['seed=63'], ['seed=123456'],
                           ['rununtiltime=1e4', 'seed=494727']])
def test1(args):
    prototype(args)

timed = timeit(prototype)


def bench1():
    timed(['seed=42', 'rununtiltime=1e8'])

import time
from math import log, exp
from random import randint

import numpy as np

from test import *  # noqa
import mtpy
import stella
from . import virtnet_utils

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
    def __eq__(self, other):
        return ((self.observations == other.observations).all() and
                self.obs_i == other.obs_i and
                self.t == other.t and
                self.next_obs_time == other.next_obs_time)

    def __str__(self):
        return "{}:{}>".format(super().__str__()[:-1], self.observations)


def uniform():
    return mtpy.mt_drand()


def mtpy_exp(p):
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
        sp.t += mtpy_exp(R)

        while isNextObservation(sp):
            makeObservation(sp)

        step(sp)


class Settings(virtnet_utils.Settings):
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
    sp_py = Spider(s, py)
    run(sp_py)

    st = np.zeros(shape=s['K'], dtype=int)
    sp_st = Spider(s, st)
    stella.wrap(run)(sp_st)

    assert id(sp_py.observations) != id(sp_st.observations)
    assert sp_py == sp_st


def prepare(params):
    sp_py = Spider(params, np.zeros(shape=params['K'], dtype=int))
    return (run, (sp_py, ), result)

def result(sp):
    return sp.observations


@mark.parametrize('args', [['seed=42'], ['seed=63'], ['seed=123456'],
                           ['rununtiltime=1e4', 'seed=494727'],
                           ['seed={}'.format(randint(1, 100000))]])
def test1(args):
    prototype(args)

timed = timeit(prototype)


def bench1():
    timed(['seed=42', 'rununtiltime=1e8'])

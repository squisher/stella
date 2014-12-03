"""
Semi-infinite 1D strip with a single spider.
"""

import mtpy  # cython wrapper around mtwist
from math import log, exp
import time
from numpy import zeros
from random import randint

from test import *  # noqa
import stella


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
            # if isinstance(type_,  FunctionType):
            #    continue
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


def mtpy_exp(p):
    u = 1.0 - mtpy.mt_drand()
    return -log(u) / p


class Simulation(object):
    EXPSTART = 0.2

    def __init__(self, params):
        self.K = params['K']
        self.rununtiltime = params['rununtiltime']
        mtpy.mt_seed32new(params['seed'])
        self.koffp = params['koffp']
        self.kcat = params['r']

        self.delta = (
            log(self.rununtiltime) - log(self.EXPSTART)) / float(self.K - 1)
        self.leg = 0
        self.substrate = 0
        self.obs_i = 0
        self.observations = zeros(shape=self.K, dtype=int)
        self.t = 0.0  # FIXME Added so that Stella knows about the member ahead of time
        self.next_obs_time = 0.0  # FIXME Added so that Stella knows about the member ahead of time

    def __str__(self):
        return "{}:{}>".format(super().__str__()[:-1], self.observations)

    def __eq__(self, o):
        assert isinstance(o, self.__class__)
        return (self.observations == o.observations).all()

    def makeObservation(self):
        """Called from run()"""
        self.observations[self.obs_i] = self.leg
        self.obs_i += 1

        self.next_obs_time = self.getNextObsTime()

    def getNextObsTime(self):
        """Called from run()"""
        if self.obs_i == 0:
            return self.EXPSTART
        if self.obs_i == self.K - 1:
            return self.rununtiltime

        return exp(log(self.EXPSTART) + self.delta * self.obs_i)

    def step(self):
        """Called from run()"""
        if self.leg == 0:
            self.leg += 1
        else:
            u1 = mtpy.mt_drand()
            if u1 < 0.5:
                self.leg -= 1
            else:
                self.leg += 1
        if self.leg == self.substrate:
            self.substrate += 1

    def isNextObservation(self):
        return self.t > self.next_obs_time and self.obs_i < self.K

    def run(self):
        self.t = 0.0
        self.next_obs_time = self.getNextObsTime()

        # TODO: Declaring R here is not necessary in Python! But llvm needs a
        # it because otherwise the definition of R does not dominate the use below.
        R = 0.0

        while self.obs_i < self.K and self.t < self.rununtiltime:
            if self.leg < self.substrate:
                R = self.koffp
            else:
                R = self.kcat
            self.t += mtpy_exp(R)

            while self.isNextObservation():
                self.makeObservation()

            self.step()


def prototype(params):
    s = Settings(params)

    sim_py = Simulation(s)
    sim_py.run()

    sim_st = Simulation(s)
    stella.wrap(sim_st.run)()

    assert id(sim_py.observations) != id(sim_st.observations)
    assert sim_py == sim_st


def prepare(params):
    sim = Simulation(params)
    return (sim.run, (), lambda: sim.observations)


@mark.parametrize('args', [['seed=42'], ['seed=63'], ['seed=123456'],
                           ['rununtiltime=1e4', 'seed=494727'],
                           ['seed={}'.format(randint(1, 100000))]])
def test1(args):
    prototype(args)


timed = timeit(prototype)


def bench1():
    timed(['seed=42', 'rununtiltime=1e8'])

# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Semi-infinite 1D strip with a single spider.
"""

import mtpy  # cython wrapper around mtwist
from math import log, exp
import time
from numpy import zeros
from random import randint

try:
    #from . import *  # noqa
    from . import mark, unimplemented
    parametrize = mark.parametrize
    from . import virtnet_utils
    import stella
except SystemError:
    def unimplemented(f):
        return f
    def parametrize(*args):
        return unimplemented
    import virtnet_utils


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
        #R = 0.0

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


def prepare(args):
    params = Settings([k+'='+str(v) for k, v in args.items()])
    sim = Simulation(params)

    def get_results(r):
        print (sim.observations)
        return sim.observations

    return (sim.run, (), get_results)


@parametrize('args', [['seed=42'], ['seed=63'], ['seed=123456'],
                           ['rununtiltime=1e4', 'seed=494727'],
                           ['seed={}'.format(randint(1, 100000))]])
def test1(args):
    prototype(args)


def main(args, wrapper=lambda x: x):
    settings = Settings(args)
    sim = Simulation(settings)
    r = wrapper(sim.run)()
    print (sim.observations)
    return r


if __name__ == '__main__':
    import sys
    main(sys.argv)

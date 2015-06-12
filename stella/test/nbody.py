#!/usr/bin/env python3
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
# The Computer Language Benchmarks Game
# http://benchmarksgame.alioth.debian.org/
#
# originally by Kevin Carson
# modified by Tupteq, Fredrik Johansson, and Daniel Nanz
# modified by Maciej Fijalkowski
# 2to3
# modified by David Mohr
#
##
# This is a specific instance of the Open Source Initiative (OSI) BSD license
# template: http://www.opensource.org/licenses/bsd-license.php
##
# Revised BSD license
#
# Copyright © 2004-2008 Brent Fulgham, 2005-2015 Isaac Gouy, 2015 David Mohr
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of "The Computer Language Benchmarks Game" nor the name of
#   "The Computer Language Shootout Benchmarks" nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import sys
import copy
import math
try:
    from . import mark, unimplemented
    parametrize = mark.parametrize
except SystemError:
    def unimplemented(f):
        return f
    def parametrize(*args):
        return unimplemented


PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

DELTA = 0.0000001


class Body(object):
    def __init__(self, p, v, mass):
        (x, y, z) = p
        (vx, vy, vz) = v
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

    def __repr__(self):
        return "Body([{},{},{}]->[{},{},{}]@{})".format(self.x, self.y, self.z,
                                                        self.vx, self.vy, self.vz,
                                                        self.mass)

    def diff(self, o):
        for a in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']:
            me = getattr(self, a)
            it = getattr(o, a)
            if abs(me - it) >= DELTA:
                raise Exception('{}: {} - {} = {} > {}'.format(a, me, it, me - it, DELTA))


class BODIES(object):
    sun = Body([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS)

    jupiter = Body([4.84143144246472090e+00,
                    -1.16032004402742839e+00,
                    -1.03622044471123109e-01],
                   [1.66007664274403694e-03 * DAYS_PER_YEAR,
                    7.69901118419740425e-03 * DAYS_PER_YEAR,
                    -6.90460016972063023e-05 * DAYS_PER_YEAR],
                   9.54791938424326609e-04 * SOLAR_MASS)

    saturn = Body([8.34336671824457987e+00,
                   4.12479856412430479e+00,
                   -4.03523417114321381e-01],
                  [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                   4.99852801234917238e-03 * DAYS_PER_YEAR,
                   2.30417297573763929e-05 * DAYS_PER_YEAR],
                  2.85885980666130812e-04 * SOLAR_MASS)

    uranus = Body([1.28943695621391310e+01,
                   -1.51111514016986312e+01,
                   -2.23307578892655734e-01],
                  [2.96460137564761618e-03 * DAYS_PER_YEAR,
                   2.37847173959480950e-03 * DAYS_PER_YEAR,
                   -2.96589568540237556e-05 * DAYS_PER_YEAR],
                  4.36624404335156298e-05 * SOLAR_MASS)

    neptune = Body([1.53796971148509165e+01,
                    -2.59193146099879641e+01,
                    1.79258772950371181e-01],
                   [2.68067772490389322e-03 * DAYS_PER_YEAR,
                    1.62824170038242295e-03 * DAYS_PER_YEAR,
                    -9.51592254519715870e-05 * DAYS_PER_YEAR],
                   5.15138902046611451e-05 * SOLAR_MASS)


SYSTEM = [BODIES.sun, BODIES.jupiter, BODIES.saturn, BODIES.uranus,
          BODIES.neptune]


def advance(dt, n, bodies):
    for i in range(n):
        for j in range(len(bodies)):
            m = j+1  # TODO workaround because no expression is supported as range arguments
            for k in range(m, len(bodies)):
                dx = bodies[j].x - bodies[k].x
                dy = bodies[j].y - bodies[k].y
                dz = bodies[j].z - bodies[k].z

                # This is extremely slow because of pow (**)
                # mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                mag = dt / (dist * dist * dist)

                b1m = bodies[j].mass * mag
                b2m = bodies[k].mass * mag
                bodies[j].vx -= dx * b2m
                bodies[j].vy -= dy * b2m
                bodies[j].vz -= dz * b2m
                bodies[k].vx += dx * b1m
                bodies[k].vy += dy * b1m
                bodies[k].vz += dz * b1m
        for j in range(len(bodies)):
            bodies[j].x += dt * bodies[j].vx
            bodies[j].y += dt * bodies[j].vy
            bodies[j].z += dt * bodies[j].vz


def advance1(dt, n, bodies):
    """Does not crash. Did not update bodies[] either!"""
    for i in range(n):
        for j in range(len(bodies)):
            m = j+1  # TODO workaround because no expression is supported as range arguments
            for k in range(m, len(bodies)):
                dx = bodies[j].x - bodies[k].x
                dy = bodies[j].y - bodies[k].y
                dz = bodies[j].z - bodies[k].z
                mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                b1m = bodies[j].mass * mag
                b2m = bodies[k].mass * mag
                bodies[j].vx -= dx * b2m
                bodies[j].vy -= dy * b2m
                bodies[j].vz -= dz * b2m
                bodies[k].vx += dx * b1m
                bodies[k].vy += dy * b1m
                bodies[k].vz += dz * b1m


def advance2(dt, n, bodies):
    """Does not crash. Did not update bodies[] either because j was initialized incorrectly."""
    for i in range(n):
        bodies[1].x = i
        for j in range(len(bodies)):
            bodies[2].x = j
            m = j+1  # TODO workaround because no expression is supported as range arguments
            for k in range(m, len(bodies)):
                bodies[3].x = 1
                bodies[j].vx -= k
                bodies[j].vy -= k
                bodies[j].vz -= k
                bodies[k].vx += k
                bodies[k].vy += k
                bodies[k].vz += k


def advance3(dt, n, bodies):
    for i in range(n):
        for j in range(len(bodies)):
            bodies[j].x += dt * bodies[j].vx
            bodies[j].y += dt * bodies[j].vy
            bodies[j].z += dt * bodies[j].vz


def calculate_energy(bodies, e=0.0):
    for j in range(len(bodies)):
        m = j+1  # TODO workaround because no expression is supported as range arguments
        for k in range(m, len(bodies)):
            dx = bodies[j].x - bodies[k].x
            dy = bodies[j].y - bodies[k].y
            dz = bodies[j].z - bodies[k].z
            e -= ((bodies[j].mass * bodies[k].mass) /
                  ((dx * dx + dy * dy + dz * dz) ** 0.5))
    for i in range(len(bodies)):
        e += bodies[i].mass * (bodies[i].vx * bodies[i].vx +
                               bodies[i].vy * bodies[i].vy +
                               bodies[i].vz * bodies[i].vz) / 2.
    return e


def report_energy(bodies, e=0.0):
    print("%.9f" % calculate_energy(bodies, e))


def offset_momentum(ref, bodies, px=0.0, py=0.0, pz=0.0):
    for i in range(len(bodies)):
        px -= bodies[i].vx * bodies[i].mass
        py -= bodies[i].vy * bodies[i].mass
        pz -= bodies[i].vz * bodies[i].mass
    ref.vx = px / ref.mass
    ref.vy = py / ref.mass
    ref.vz = pz / ref.mass


@parametrize('opt', [3, 2, 1])
def test1a(opt):
    return _test1(opt)


@parametrize('opt', [0])
@unimplemented
def test1b(opt):
    """At -O0 the alloca of the nested for loops will cause a stack overflow."""
    return _test1(opt)


def _test1(opt):
    import stella
    n = 5990
    offset_momentum(BODIES.sun, SYSTEM)
    s1 = copy.deepcopy(SYSTEM)
    s2 = copy.deepcopy(SYSTEM)
    r1 = advance(0.01, n, s1)
    r2 = stella.wrap(advance, opt=opt)(0.01, n, s2)

    for i, body in enumerate(s1):
        body.diff(s2[i])

    assert r1 == r2 and abs(calculate_energy(s1) - calculate_energy(s2)) < DELTA


def format_e(dt, n, system):
    return "{:.9f}".format(calculate_energy(system))


def prepare(args):
    system = init()
    report_energy(system)
    return (advance, (args['dt'], args['n'], system), format_e)


def init():
    system = copy.deepcopy(SYSTEM)
    offset_momentum(system[0], system)
    return system


def main(n, wrapper=lambda x: x):
    system = init()
    report_energy(system)
    r = wrapper(advance)(0.01, n, system)
    report_energy(system)
    return r


if __name__ == '__main__':
    main(int(sys.argv[1]))

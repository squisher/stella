#!/usr/bin/env python3
# The Computer Language Benchmarks Game
# http://benchmarksgame.alioth.debian.org/
#
# originally by Kevin Carson
# modified by Tupteq, Fredrik Johansson, and Daniel Nanz
# modified by Maciej Fijalkowski
# 2to3
# modified by David Mohr


import sys
import copy
import math
import stella


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


def test1():
    n = 5000
    offset_momentum(BODIES.sun, SYSTEM)
    s1 = copy.deepcopy(SYSTEM)
    s2 = copy.deepcopy(SYSTEM)
    r1 = advance1(0.01, n, s1)
    r2 = stella.wrap(advance1)(0.01, n, s2)

    for i, body in enumerate(s1):
        body.diff(s2[i])

    assert r1 == r2 and abs(calculate_energy(s1) - calculate_energy(s2)) < DELTA


def format_e(dt, n, system):
    return calculate_energy(system)


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

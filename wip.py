import stella.test.si1l1s_struct
import stella.test.basicmath
import stella.test.langconstr
import stella.test.objects
import stella.test.external_func
import stella.test.si1l1s_struct
import stella.test.si1l1s_obj
import stella.test.virtnet_purepython
import stella.test.nbody
import stella.test.errors
import stella.test.heat
import stella.test.typing
import stella
from stella import exc  # noqa
import numpy as np
import mtpy  # noqa
import ctypes  # noqa


# a = np.zeros(5, dtype=int)
a = np.array([1, 2, 3, 4])
b = stella.test.objects.B()
b2 = stella.test.objects.B(0.0, 1.0)
b3 = stella.test.objects.B(0.0, 1.0)
b3.next = b3
c = stella.test.objects.C(np.array([1, 2, 3, 42]))
d = np.zeros((2, 2), dtype=int)
dd = np.array([[4.2, 3], [2.0, -1]])
ddd = np.array([[1.5,  2.5,  5.5], [-3.3, -5.7,  1.1]])
ddb = np.zeros(ddd.shape)
e = stella.test.objects.E()
e2 = stella.test.objects.E()
settings1d = stella.test.si1l1s_obj.Settings(['seed=42'])
sim1d = stella.test.si1l1s_obj.Simulation(settings1d)
settings = stella.test.virtnet_purepython.Settings(['seed=42'])
sim = stella.test.virtnet_purepython.Simulation(settings)
l1 = [stella.test.objects.E(2), stella.test.objects.E(4)]
f = stella.test.objects.F(l1)
g = stella.test.objects.G(2, 9)
h = stella.test.objects.H(9, 7, 3)
j = stella.test.objects.J((-1, 2))
heat = stella.test.heat.Sim()
stella.test.heat.process_config(heat, 'test1_settings.txt')


def ret(o):
    o.i = 667
    return o


def current_work(run=False, **kwargs):
    if type(run) == bool:
        ir = not run
    else:
        ir = run
    print(stella.wrap(stella.test.basicmath.power2, ir=ir, **kwargs)(2, 2))

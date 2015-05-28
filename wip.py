import test.si1l1s_struct
import test.basicmath
import test.langconstr
import test.objects
import test.external_func
import test.si1l1s_struct
import test.si1l1s_obj
import test.virtnet_purepython
import test.nbody
import test.errors
import test.heat
import test.typing
import stella
from stella import exc  # noqa
import numpy as np
import mtpy  # noqa
import ctypes  # noqa


# a = np.zeros(5, dtype=int)
a = np.array([1, 2, 3, 4])
b = test.objects.B()
b2 = test.objects.B(0.0, 1.0)
b3 = test.objects.B(0.0, 1.0)
b3.next = b3
c = test.objects.C(np.array([1, 2, 3, 42]))
d = np.zeros((2, 2), dtype=int)
dd = np.array([[4.2, 3], [2.0, -1]])
ddd = np.array([[1.5,  2.5,  5.5], [-3.3, -5.7,  1.1]])
ddb = np.zeros(ddd.shape)
e = test.objects.E()
e2 = test.objects.E()
settings = test.virtnet_purepython.Settings(['seed=42'])
sim = test.virtnet_purepython.Simulation(settings)
l1 = [test.objects.E(2), test.objects.E(4)]
f = test.objects.F(l1)
g = test.objects.G(2, 9)
h = test.objects.H(9, 7, 3)
j = test.objects.J((-1, 2))
heat = test.heat.Sim()
test.heat.process_config(heat, 'test1_settings.txt')


def ret(o):
    o.i = 667
    return o


def current_work(run=False, **kwargs):
    if type(run) == bool:
        ir = not run
    else:
        ir = run
    print(stella.wrap(test.typing.cast_bool, ir=ir, **kwargs)(5))

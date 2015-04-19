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
import stella
from stella import exc  # noqa
import numpy as np
import mtpy  # noqa
import ctypes  # noqa


# a = np.zeros(5, dtype=int)
a = np.array([1, 2, 3])
b = test.objects.B()
b2 = test.objects.B(0.0, 1.0)
b3 = test.objects.B(0.0, 1.0)
b3.next = b3
c = test.objects.C(np.array([1, 2, 3, 42]))
e = test.objects.E()
e2 = test.objects.E()
settings = test.virtnet_purepython.Settings(['seed=42'])
sim = test.virtnet_purepython.Simulation(settings)
l1 = [test.objects.E(2), test.objects.E(4)]
f = test.objects.F(l1)
g = test.objects.G(2, 9)
h = test.objects.H(9, 7, 3)


def current_work(run=False, **kwargs):
    if type(run) == bool:
        ir = not run
    else:
        ir = run
    print(stella.wrap(test.objects.selfRef, ir=ir, **kwargs)(g))

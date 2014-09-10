import test.si1l1s_struct
import test.basicmath
import test.langconstr
import test.objects
import test.si1l1s_struct
import stella
from stella import exc  # noqa
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))
import numpy as np
import mtpy  # noqa
import ctypes  # noqa

a = np.zeros(5, dtype=int)
b = test.objects.B()
b2 = test.objects.B(0.0, 1.0)
c = test.objects.C(np.array([1,2,3,42]))
settings = test.si1l1s_struct.Settings(['seed=42'])
sp = test.si1l1s_struct.Spider(settings, np.zeros(shape=settings['K'], dtype=int))

def current_work(run=False):
    print(stella.wrap(test.langconstr.and_, ir=not run)(True, True))

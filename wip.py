import test.si1l1s_struct
import test.basicmath
import test.langconstr
import test.objects
import stella
from stella import exc
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))
import numpy as np
import mtpy

a = np.zeros(5, dtype=int)
b = test.objects.B()
b2 = test.objects.B(0.0, 1.0)
c = test.objects.C()

def current_work(run=False):
    print(c)
    print(stella.wrap(test.objects.getArrayValue, ir=not run)(c))
    if run:
        print(c)

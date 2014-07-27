import test.si1l1s
import test.basicmath
import test.langconstr
import test.struct
import stella
from stella import exc
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))
import numpy as np
import mtpy

a = np.zeros(5, dtype=int)
b = test.struct.B()
b2 = test.struct.B(0.0, 1.0)

def current_work(run=False):
    print(b)
    print(stella.wrap(test.struct.setAttribFloat, ir=not run)(b))
    if run:
        print(b)

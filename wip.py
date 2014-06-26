import test.si1l1s
import test.basicmath
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))
import numpy as np
import mtpy

a = np.zeros(5, dtype=int)

def current_work(run=False):
    #print(stella.wrap(exponential, ir=not run)(42))
    test.si1l1s.bench1()

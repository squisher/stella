from test.external_func import *
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))
import numpy as np
import mtpy

a = np.zeros(5, dtype=int)

def current_work(run=False):
    print(stella.wrap(seed, ir=not run)())

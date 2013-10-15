from stella import stella
import sys

def single_args(l):
    return list(map (lambda x: (x,), l))

def make_eq_test(name, f, arglist):
    def perform_eq_test(f, *args):
        x = f(*args)
        y = stella(f)(*args)
        assert x == y and type(x) == type(y)
    for i in range(len(arglist)):
        sys.modules[name].__dict__["test_eq_" + f.__name__+"_"+str(i)] = lambda: perform_eq_test(f, *arglist[i])
    if len(arglist) == 0:
        sys.modules[name].__dict__["test_eq_" + f.__name__] = lambda: perform_eq_test(f)

def make_delta_test(name, f, arglist, delta = 0.0000001):
    def perform_delta_test(f, delta, *args):
        x = f(*args)
        y = stella(f)(*args)
        assert x-y < delta and type(x) == type(y)
    for i in range(len(arglist)):
        sys.modules[name].__dict__["test_delta_" + f.__name__+"_"+str(i)] = lambda: perform_delta_test(f, delta, *arglist[i])
    if len(arglist) == 0:
        sys.modules[name].__dict__["test_delta_" + f.__name__] = lambda: perform_delta_test(f, delta)

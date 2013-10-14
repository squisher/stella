from stella import stella
import sys

def make_eq_test(name, f, arglist):
    for i in range(len(arglist)):
        def perform_eq_test(f, args):
            a, b = args
            x = f(a,b)
            y = stella(f)(a,b)
            assert x == y and type(x) == type(y)
        sys.modules[name].__dict__["test_" + f.__name__+"_"+str(i)] = lambda: perform_eq_test(f, arglist[i])

def make_delta_test(name, f, arglist, delta = 0.0000001):
    for i in range(len(arglist)):
        def perform_delta_test(f, args):
            a, b = args
            x = f(a,b)
            y = stella(f)(a,b)
            assert x-y < delta and type(x) == type(y)
        sys.modules[name].__dict__["test_" + f.__name__+"_"+str(i)] = lambda: perform_delta_test(f, arglist[i])

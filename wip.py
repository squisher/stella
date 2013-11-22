def f(a,b): return a and b
#       select on %a, %r, %a, %a  // assign a to r
#       jump :phi
#       select on %b, %r, %b, %b  // assign b to r
#       phi %r
#       return %r
def g(a):
    if a:
        return 1
    else:
        return 2

from test.basicmath import *
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))

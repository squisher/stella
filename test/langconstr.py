from test import *

def direct_assignment(x,y):
    a = x
    return a + y

def simple_assignment(x,y):
    a = x + y
    return a

for f in [direct_assignment, simple_assignment]:
    make_eq_test(__name__, f, [(40,2), (42.0, 0)])

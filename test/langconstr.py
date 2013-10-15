from test import *

def direct_assignment(x,y):
    a = x
    return a + y

def simple_assignment(x,y):
    a = x + y
    return a

def double_assignment(x,y):
    a = x
    b = 5 + y
    a += b
    return a

def double_cast(x,y):
    a = x / y
    b = y // x
    return a + b

for f in [direct_assignment, simple_assignment, double_assignment, double_cast]:
    make_eq_test(__name__, f, [(40,2), (42.0, 0), (41, 1.0)])

if __name__ == '__main__':
    print(stella(double_cast, debug='print')(0, 42))

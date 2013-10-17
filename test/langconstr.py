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

@mark.parametrize('f', [direct_assignment, simple_assignment, double_assignment, double_cast])
@mark.parametrize('args', [(40,2), (42.0, 0), (41, 1.0)])
def test1(f,args):
    make_eq_test(f, args)

if __name__ == '__main__':
    print(stella(double_cast, debug='print')(0, 42))

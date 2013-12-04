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

def and_(a,b): return a and b
def or_(a,b):  return a or b

@mark.parametrize('args', [(40,2), (43.0, -1), (41, 1.0)])
@mark.parametrize('f', [direct_assignment, simple_assignment, double_assignment, double_cast])
def test1(f,args):
    make_eq_test(f, args)

@mark.parametrize('args', [(True, True), (True, False), (False, True), (False, False)])
@mark.parametrize('f', [and_, or_])
def test2(f,args):
    make_eq_test(f, args)

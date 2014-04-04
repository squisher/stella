from test import *

def direct_assignment(x,y):
    a = x
    return a + y

def simple_assignment(x,y):
    a = x + y
    return a

def return_const():
    return 41

def assign_const():
    r = 42
    return r

def double_assignment(x,y):
    a = x
    b = 5 + y
    a += b
    return a

def double_cast(x,y):
    a = x / y
    b = y // x
    return a + b

def simple_if(x):
    if x:
        return 0
    else:
        return 42

def simple_ifeq(x,y):
    if x==y:
        return 0
    else:
        return 42

def simple_ifeq_const(x):
    if x==False:
        return 0
    else:
        return 42

def for1(x):
    r = 0
    for i in range(x):
        r += i
    return r

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

@mark.parametrize('arg', single_args([True, False]))
@mark.parametrize('f', [simple_if, simple_ifeq_const])
def test3(f,arg):
    make_eq_test(f, arg)

@mark.parametrize('args', [(True, False), (True, True), (4, 2), (4.0, 4.0)])
@mark.parametrize('f', [simple_ifeq])
def test4(f,args):
    make_eq_test(f, args)

@mark.parametrize('f', [return_const, assign_const])
def test5(f):
    make_eq_test(f, ())

@mark.parametrize('arg', single_args([0, 1, 42]))
@mark.parametrize('f', [for1])
def test6(f,arg):
    make_eq_test(f, arg)


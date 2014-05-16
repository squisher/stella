from test import *
from stella import zeros
from .basicmath import addition, subtraction

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

def for2(x):
    r = 0
    s = 1
    for i in range(x):
        r += i
        s *= 2
    return r+s

def while1(x):
    r = 0
    while x>0:
        r += x
        x -= 1
    return r

def recursive(x):
    if x <= 0:
        return 1
    else:
        return x+recursive(x-1)

def fib(x):
    if x <= 2:
        return 1
    return fib(x-1) + fib (x-2)

def fib_harness(n, x):
    """
    Test calling an external function.
    """
    r = 0
    for i in range(n):
        r += fib(x)
    return r

def hof_f(n):
    if n == 0: return 1
    else:      return n - hof_m(hof_f(n-1))

def hof_m(n):
    if n == 0: return 0
    else:      return n - hof_f(hof_m(n-1))

def and_(a,b): return a and b
def or_(a,b):  return a or b

some_global = 0
def use_global():
    global some_global
    some_global = 0
    x = 5
    while some_global == 0:
        x = global_test_worker(x)
    return x

def global_test_worker(x):
    global some_global
    if x < 0:
        some_global = 1
    return x-1

def kwargs(a=0, b=1):
    return a+b

def call_return(x,y):
    if y > 0:
        return addition(x,y)
    else:
        return subtraction(x,y)

def array_allocation():
    a = zeros(5, dtype=int)

def array_alloc_assignment():
    a = zeros(5)
    for i in range(5):
        a[i] = i
    return i

def array_alloc_use():
    a = zeros(5)
    for i in range(5):
        a[i] = i**2
    r = 0
    for i in range(5):
        r += a[i]
    return r

def array_len():
    """ TODO: is there a reason not to support len? """
    a = zeros(5)
    return len(a)


@mark.parametrize('args', [(40,2), (43, -1), (41, 1)])
@mark.parametrize('f', [direct_assignment, simple_assignment, double_assignment, double_cast, call_return])
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

@mark.parametrize('f', [return_const, assign_const, use_global])
def test5(f):
    make_eq_test(f, ())

@mark.parametrize('f', [array_allocation, array_alloc_assignment, array_alloc_use, array_len])
@unimplemented
def test5b(f):
    make_eq_test(f, ())

@mark.parametrize('arg', single_args([0, 1, 42, -1, -42]))
@mark.parametrize('f', [for1, for2, while1, recursive])
def test6(f,arg):
    make_eq_test(f, arg)

@mark.parametrize('arg', single_args([0, 1, 2, 5, 8, -1, -3]))
@mark.parametrize('f', [fib])
def test7(f,arg):
    make_eq_test(f, arg)

@mark.parametrize('f', [fib_harness, kwargs])
def test8(f):
    make_eq_test(f, (1,30))

@mark.parametrize('arg', single_args([0, 1, 2, 5, 8, 12]))
@mark.parametrize('f', [hof_f])
def test9(f,arg):
    make_eq_test(f, arg)

@mark.parametrize('args', [{'a':1}, {'b':2}, {'a':1, 'b':0}, {'b':1, 'a':0}, {'a':1.2}, {'b':-3}, {}])
def test10(args):
    make_eq_kw_test(kwargs, args)

@mark.parametrize('args', [{'c': 5}, {'b': -1, 'c': 5}])
@mark.xfail()
def test11(args):
    make_eq_kw_test(kwargs, args)

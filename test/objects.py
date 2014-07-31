import numpy as np

from test import *  # noqa
import stella
from stella import exc

class B(object):
    x = 0
    y = 0

    def __init__(self, x=1, y=2):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}:{}, {}>".format(str(type(self))[:-1], self.x, self.y)


class C(object):
    """
    %"class 'test.objects.C'_<Int*6>_Int" = type { [6 x i64]*, i64 }
    """
    def __init__(self, obj, i=0):
        if isinstance(obj, int):
            self.a = np.zeros(shape=obj, dtype=int)
            self.a[0] = 42
        else:
            self.a = np.array(obj)
        self.i = i

    def __eq__(self, other):
        return self.i == other.i and (self.a == other.a).all()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}:{}, {}>".format(str(type(self))[:-1], self.i, self.a)


def justPassing(a):
    x = 1


def cmpAttrib(a):
    return a.x == a.y


def setAttrib(a):
    a.x = 42


def setAttribFloat(a):
    a.x = 42.0


def setUnknownAttrib(a):
    a.z = 42


def getAttrib(a):
    return a.x


def addAttribs(a):
    return a.x + a.y


def returnUnknownAttrib(a):
    return a.z


def getAndSetAttrib1(a):
    a.x *= a.y


def getAndSetAttrib2(a):
    a.x -= 1


args1 = [(1,1), (24, 42), (0.0, 1.0), (1.0, 1.0), (3.0, 0.0)]

def getFirstArrayValue(c):
    return c.a[0]


def getSomeArrayValue(c, i):
    return c.a[i]


def sumC(c):
    for i in range(len(c.a)):
        c.i += c.a[i]


@mark.parametrize('f', [justPassing, addAttribs, getAttrib])
@mark.parametrize('args', args1)
def test_no_mutation(f, args):
    b1 = B(*args)
    b2 = B(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('f', [])
@unimplemented
def test_no_mutation_u(f):
    b1 = B()
    b2 = B()

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

@mark.parametrize('f', [setAttrib])
def test_mutation(f):
    b1 = B()
    b2 = B()

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 != B() and b1 == b2 and py == st


@mark.parametrize('f', [setAttribFloat])
@mark.xfail(raises=TypeError)
def test_mutation_f(f):
    """
    The opposite, setting an int when the struct member is float does not
    raise a TypeError since the int will be promoted to a float.
    """
    b1 = B()
    b2 = B()

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 != B() and b1 == b2 and py == st


@mark.parametrize('args', args1)
@mark.parametrize('f', [cmpAttrib, getAndSetAttrib1, getAndSetAttrib2])
def test_mutation2(f, args):
    b1 = B(*args)
    b2 = B(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('args', [])
@mark.parametrize('f', [])
@unimplemented
def test_mutation2_u(f, args):
    b1 = B(*args)
    b2 = B(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

@mark.parametrize('f', [returnUnknownAttrib])
@mark.xfail(raises=AttributeError)
def test_mutation2_f(f):
    b1 = B()
    b2 = B()

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


args2 = [(1,2,3,4), (1.0, 2.0, 3.0)]
args3 = list(zip(args2, [0, 0.0]))


@mark.parametrize('f', [getFirstArrayValue, sumC])
@mark.parametrize('args', args3)
def test_no_mutation2(f, args):
    b1 = C(*args)
    b2 = C(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('f', [])
@mark.parametrize('args', args2)
@unimplemented
def test_no_mutation2_f(f, args):
    b1 = C(args)
    b2 = C(args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('f', [getSomeArrayValue])
@mark.parametrize('args', args2)
def test_no_mutation3(f, args):
    b1 = C(args)
    b2 = C(args)

    assert b1 == b2
    py = f(b1, 1)
    st = stella.wrap(f)(b2, 1)

    assert b1 == b2 and py == st

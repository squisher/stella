from test import *  # noqa
import stella

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


def justPassing(a):
    x = 1


def cmpAttrib(a):
    return a.x == a.y


def setAttrib(a):
    a.x += 1


def addAttribs(a):
    return a.x + a.y


@mark.parametrize('f', [justPassing, addAttribs])
def test_no_mutation(f):
    b1 = B()
    b2 = B()

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


@mark.parametrize('args', [(1,1), (24, 42), (0.0, 1.0), (1.0, 1.0), (3.0, 0.0)])
@mark.parametrize('f', [cmpAttrib])
def test_mutation2(f, args):
    b1 = B(*args)
    b2 = B(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('args', [])
@mark.parametrize('f', [cmpAttrib])
@unimplemented
def test_mutation2_u(f, args):
    b1 = B(*args)
    b2 = B(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

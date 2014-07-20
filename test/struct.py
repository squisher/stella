from test import *  # noqa

class B(object):
    x = 1
    y = 2

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)


def justPassing(a):
    x = 1


def setAttrib(a):
    a.x += 1


def addAttribs(a):
    return a.x + a.y


@mark.parametrize('f', [justPassing])
def test1(f):
    b1 = B()
    b2 = B()

    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

@mark.parametrize('f', [setAttrib, addAttribs, ])
@unimplemented
def test1(f):
    b1 = B()
    b2 = B()

    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

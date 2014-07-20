from test import *  # noqa

class B(object):
    x = 1
    y = 2

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)


def setAttrib(a):
    a.x += 1


@unimplemented
def test1():
    b1 = B()
    b2 = B()

    py = setAttrib(a)
    st = stella.wrap(setAttrib)(a)

    assert b1 == b2 and py == st

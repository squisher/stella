import numpy as np

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


class D(object):
    z = 0
    a = 0
    y = 0.0
    g = 0.0

    def __init__(self):
        pass

    def __eq__(self, other):
        return (self.z == other.z and
                self.a == other.a and
                self.y == other.y and
                self.g == other.g)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}: {}>".format(str(type(self))[:-1], [self.z, self.a, self.y, self.g])


class E(object):
    def __init__(self, x=0):
        self.x = x

    def inc(self, p=1):
        self.x += p
        return self.x

    def __eq__(self, other):
        return (self.x == other.x)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}[x={}]".format(str(type(self))[8:-2], self.x)


class F(object):
    def __init__(self, l):
        self.l = l

    def __eq__(self, other):
        return (self.l == other.l)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}[l={}]".format(str(type(self))[8:-2], self.l)


class G(B):
    def __init__(self, x=1, y=2):
        super().__init__(x, y)


G.origin = G(0, 0)


def justPassing(a):
    x = 1  # noqa


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


def callBoundMethod(e):
    e.inc()
    return e.x


def callBoundMethod2(e):
    e.inc(42)


def callBoundMethod3(e, x):
    e.inc(x)


def callBoundMethodTwice(e, x):
    e.inc(x)
    e.inc(x)


def callBoundMethodOnTwo(e1, e2):
    e1.inc(1)
    e2.inc(2)


def objList1(l):
    return l[0].x + l[1].x


def objList2(l):
    r = 0
    for i in range(len(l)):
        r += l[i].x
    return r


def objList3(l):
    r = 0
    for i in range(len(l)):
        for j in range(len(l)):
            r += l[j].x + i
    return r


def objList4(l):
    for i in range(len(l)):
        l[i].x = i


def objContainingList1(f):
    return f.l[0].x + f.l[1].x


def objContainingList2(f):
    r = 0
    for i in range(len(f.l)):
        r += f.l[i].x
    return r


def objContainingList3(f):
    for i in range(len(f.l)):
        f.l[i].x = i


def selfRef(g):
    return ((g.x - G.origin.x)**2 + (g.y - G.origin.y)**2)**0.5


def nextB(b):
    return b.x == b.next.x and b.y == b.next.y


args1 = [(1, 1), (24, 42), (0.0, 1.0), (1.0, 1.0), (3.0, 0.0)]


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


args2 = [(1, 2, 3, 4), (1.0, 2.0, 3.0)]
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


def manipulate_d1(d):
    d.z = 1
    d.a = 2
    d.y = 3.0
    d.g = 4.0


def pass_struct(d):
    manipulate_d1(d)


@mark.parametrize('f', [manipulate_d1, pass_struct])
def test_mutation3(f):
    b1 = D()
    b2 = D()

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('f', [callBoundMethod, callBoundMethod2])
def test_mutation4(f):
    e1 = E()
    e2 = E()

    assert e1 == e2

    py = f(e1)
    st = stella.wrap(f)(e2)

    assert e1 == e2 and py == st


@mark.parametrize('f', [callBoundMethod3, callBoundMethodTwice])
@mark.parametrize('arg', [0, -1, 5])
def test_mutation5(f, arg):
    e1 = E()
    e2 = E()

    assert e1 == e2

    py = f(e1, arg)
    st = stella.wrap(f)(e2, arg)

    assert e1 == e2 and py == st


@mark.parametrize('f', [callBoundMethodOnTwo])
def test_mutation6(f):
    e1 = E()
    e2 = E()
    e3 = E()
    e4 = E()

    assert e1 == e2 and e3 == e4

    py = f(e1, e3)
    st = stella.wrap(f)(e2, e4)

    assert e1 == e2 and e3 == e4 and py == st


@mark.parametrize('f', [objList1, objList2, objList3])
def test_no_mutation7(f):
    l1 = [E(4), E(1)]
    l2 = [E(4), E(1)]

    py = f(l1)
    st = stella.wrap(f)(l2)

    assert l1 == l2 and py == st


@mark.parametrize('f', [objList4])
def test_mutation7(f):
    l1 = [E(4), E(1)]
    l2 = [E(4), E(1)]

    py = f(l1)
    st = stella.wrap(f)(l2)

    assert l1 == l2 and py == st


@mark.parametrize('f', [objContainingList1, objContainingList2])
def test_no_mutation8(f):
    l1 = [E(2), E(5)]
    l2 = [E(2), E(5)]
    f1 = F(l1)
    f2 = F(l2)

    py = f(f1)
    st = stella.wrap(f)(f2)

    assert f1 == f2 and py == st


@mark.parametrize('f', [objContainingList3])
def test_mutation8(f):
    l1 = [E(2), E(5)]
    l2 = [E(2), E(5)]
    f1 = F(l1)
    f2 = F(l2)

    py = f(f1)
    st = stella.wrap(f)(f2)

    assert f1 == f2 and py == st


args3 = [(4, 8), (9.0, 27.0)]

@mark.parametrize('f', [nextB])
@mark.parametrize('args', args3)
def test_no_mutation9(f, args):
    b1 = B(*args)
    b2 = B(*args)

    b1.next = b1
    b2.next = b2

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st


@mark.parametrize('f', [selfRef])
@mark.parametrize('args', args3)
@unimplemented
def test_no_mutation9_u(f, args):
    b1 = G(*args)
    b2 = G(*args)

    assert b1 == b2
    py = f(b1)
    st = stella.wrap(f)(b2)

    assert b1 == b2 and py == st

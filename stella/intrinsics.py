import sys
from abc import ABCMeta, abstractmethod, abstractproperty

from .llvm import *
from .exc import *

from functools import wraps

class Intrinsic(metaclass=ABCMeta):
    py_func = None
    @abstractmethod
    def __init__ (self, args, kw_args):
        pass
    @abstractmethod
    def getReturnType(self):
        pass
    @abstractmethod
    def translate(self, module, builder):
        pass

def zeros(shape=1, dtype=None):
    """Emulate certain features of `numpy.zeros`

    Note that `dtype` is ignored in Python, but will be interpreted in Stella.
    """
    try:
        dim = len(shape)
        if dim == 1:
            shape=shape[0]
            raise TypeError()
    except TypeError:
        return [0 for i in range(shape)]

    # here dim > 1, build up the inner most dimension
    inner = [0 for i in range(shape[dim-1])]
    for i in range(dim-2,-1,-1):
        new_inner = [list(inner) for j in range(shape[i])]
        inner = new_inner
    return inner

class Zeros(Intrinsic):
    py_func = zeros
    def __init__(self, args):
        # for now only 1D
        self.shape = args[0]
        self.type = args[1]
    def getReturnType(self):
        return tp_array(self.tp, self.n)
    def translate(self, module, builder):
        return builder.alloca(self.tp, self.n)


# --


func2klass = {}
# Get all contrete subclasses of Intrinsic and register them
for name in dir(sys.modules[__name__]):
    klass = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(klass, Intrinsic) and len(klass.__abstractmethods__) == 0 and klass.py_func != None:
            func2klass[klass.py_func] = klass
    except TypeError:
        pass

def getIntrinsic(func):
    global func2klass

    if func in func2klass:
        return func2klass[func]
    else:
        return None

def getPythonIntrinsics():
    global func2klass

    return func2klass.keys()

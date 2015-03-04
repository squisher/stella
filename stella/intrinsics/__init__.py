"""
Intrinsics
"""

import sys
import math
from abc import abstractmethod
from . import python
from .. import tp, exc
from ..storage import Register


class Intrinsic(tp.Foreign, tp.Callable):
    py_func = None
    arg_names = []
    arg_defaults = []

    def __init__(self):
        self.type_ = tp.IntrinsicType(self.py_func, self.arg_names, self.arg_defaults)

    @abstractmethod
    def call(self, cge, args, kw_args):
        """args and kw_args are already added by a call through addArgs()"""
        pass

    def getResult(self, func):
        return Register(func)


class Zeros(Intrinsic):
    py_func = python.zeros
    arg_names = ['shape', 'dtype']
    arg_defaults = [tp.PyWrapper(int)]

    def getReturnType(self, args, kw_args):
        combined = self.combineArgs(args, kw_args)
        shape = combined[0].value
        type_ = tp.get_scalar(combined[1])
        if not tp.supported_scalar(type_):
            raise exc.TypeError("Invalid array element type {0}".format(type_))
        atype = tp.ArrayType(type_, shape)
        return atype

    def call(self, cge, args, kw_args):
        type_ = self.getReturnType(args, kw_args).llvmType()
        return cge.builder.alloca(type_)


class Len(Intrinsic):
    """
    Determine the length of the array based on its type.
    """
    py_func = len
    arg_names = ['obj']

    def getReturnType(self, args, kw_args):
        return tp.Int

    def getResult(self, func):
        # we need the reference to back-patch
        self.result = tp.Const(-42)
        return self.result

    def call(self, cge, args, kw_args):
        obj = args[0]
        if obj.type.isReference():
            type_ = obj.type.dereference()
        else:
            type_ = obj.type
        if not isinstance(type_, tp.ArrayType):
            raise exc.TypeError("Invalid array type {0}".format(obj.type))
        self.result.value = type_.shape
        self.result.translate(cge)
        return self.result.llvm


class Log(Intrinsic):
    py_func = math.log
    intr = 'llvm.log'
    arg_names = ['x']  # TODO: , base

    def getReturnType(self, args, kw_args):
        return tp.Float

    def call(self, cge, args, kw_args):
        if args[0].type == tp.Int:
            args[0] = tp.Cast(args[0], tp.Float)

        # TODO llvmlite
        llvm_f = cge.module.llvm.declare_intrinsic(self.intr, [args[0].llvmType()])
        result = cge.builder.call(llvm_f, [args[0].translate(cge)])
        return result


class Exp(Log):
    py_func = math.exp
    intr = 'llvm.exp'
    arg_names = ['x']


class Sqrt(Log):
    py_func = math.sqrt
    intr = 'llvm.sqrt'
    arg_names = ['x']


func2klass = {}


# Get all contrete subclasses of Intrinsic and register them
for name in dir(sys.modules[__name__]):
    klass = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(klass, Intrinsic) and len(klass.__abstractmethods__) == 0 and \
                klass.py_func is not None:
            func2klass[klass.py_func] = klass
    except TypeError:
        pass

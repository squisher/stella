# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Intrinsics
"""

import sys
import math
from abc import abstractmethod
import builtins
import llvmlite.ir as ll

from . import python
from .. import tp, exc
from ..storage import Register
import numpy as np


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

        # the Python type when passed around in Stella is the intrinsic
        # function which is a subtype of Cast
        assert isinstance(combined[1], Cast)
        type_ = tp.get_scalar(combined[1].py_func)

        if not tp.supported_scalar(type_):
            raise exc.TypeError("Invalid array element type {0}".format(type_))
        atype = tp.ArrayType(type_, shape)
        atype.complex_on_stack = True
        atype.on_heap = False
        return atype

    def call(self, cge, args, kw_args):
        type_ = self.getReturnType(args, kw_args).llvmType(cge.module)
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
        llvm_f = cge.module.llvm.declare_intrinsic(self.intr, [args[0].llvmType(cge.module)])
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


class Exception(Intrinsic):
    """
    NoOP since Exceptions are not objects in Stella.
    """
    py_func = builtins.Exception
    arg_names = ['msg']

    def getReturnType(self, args, kw_args):
        return tp.Void

    def getResult(self, func):
        return Register(func)

    def call(self, cge, args, kw_args):
        pass

    @staticmethod
    def is_a(item):
        return isinstance(item, type) and issubclass(item, builtins.Exception)


class Cast(Intrinsic):
    """
    Abstract cast
    """
    arg_names = ['x']

    def __init__(self):
        super().__init__()

    def getReturnType(self, args, kw_args):
        return self.stella_type

    def call(self, cge, args, kw_args):
        obj = args[0]
        cast = tp.Cast(obj, self.stella_type)
        return cast.translate(cge)


class Float(Cast):
    stella_type = tp.Float
    py_func = stella_type.type_


class Int(Cast):
    stella_type = tp.Int
    py_func = stella_type.type_


class Bool(Cast):
    stella_type = tp.Bool
    py_func = stella_type.type_


class Tuple(Intrinsic):
    py_func = tuple
    arg_names = ['iterable']

    def __init__(self):
        super().__init__()

    def getReturnType(self, args, kw_args):
        type_ = args[0].type.dereference()
        assert isinstance(type_, tp.Subscriptable)
        # there are no Nd tuples, accept only 1d
        assert not isinstance(type_.shape, list)

        ttype = tp.TupleType([type_.type_] * type_.shape)
        return ttype

    def call(self, cge, args, kw_args):
        in_type = args[0].type.dereference()
        out_type = self.getReturnType(args, kw_args).llvmType(cge.module)

        init = [ll.Constant(in_type.type_.llvmType(cge.module), None)] * in_type.shape

        llvm = ll.Constant.literal_struct(init)

        for i in range(in_type.shape):
            val = in_type.loadSubscript(cge, args[0], tp.Const(i))
            llvm = cge.builder.insert_value(llvm, val, i)

        return llvm


casts = (int, float, bool, tuple)


def is_extra(item):
    """Allow more flexible intrinsics detection than simple equality.

    The example is Exception, where we want to catch subtypes as well.
    """
    # `numpy.ndarray` in `tuple` is broken, so work around it
    if isinstance(item, np.ndarray):
        return False
    return any([f(item) for f in [Exception.is_a]]) or item in casts


def get(func):
    if func in func2klass:
        return func2klass[func]
    elif Exception.is_a(func):
        return Exception
    else:
        return None


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

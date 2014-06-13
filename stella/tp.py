import logging

import llvm
import llvm.core
import llvm.ee
import numpy as np

from .exc import *

class Type(object):
    type_ = None
    _llvm = None

    def __str__(self):
        return '<?>'
    # llvm.core.ArrayType does something funny, it will compare against _ptr,
    # so let's just add the attribute here to enable equality tests
    _ptr = None

    def llvmType(self):
        raise TypingError("Cannot create llvm type for an unknown type. This should have been cought earlier.")

NoType = Type()
Void = None

class ScalarType(Type):
    def __init__(self, type_, llvm, f_generic_value, f_constant):
        self.type_ = type_
        self._llvm = llvm
        self.f_generic_value = f_generic_value
        self.f_constant = f_constant

    def llvmType(self):
        return self._llvm

    def genericValue(self, value):
        return self.f_generic_value(self._llvm, value)

    def constant(self, value):
        return self.f_constant(self._llvm, value)

    def __str__(self):
        return self.type_.__name__

tp_int = llvm.core.Type.int(64)
#tp_int32 = llvm.core.Type.int(32)
#tp_float = llvm.core.Type.float() # Python always works with double precision
tp_double = llvm.core.Type.double()
tp_bool = llvm.core.Type.int(1)
tp_void = llvm.core.Type.void()

Int = ScalarType(
    int, tp_int,
    llvm.ee.GenericValue.int_signed,
    llvm.core.Constant.int
)
Float  = ScalarType(
    float, tp_double,
    llvm.ee.GenericValue.real,
    llvm.core.Constant.real
)
Bool = ScalarType(
    bool, tp_bool,
    llvm.ee.GenericValue.int,
    llvm.core.Constant.int
)
None_ = ScalarType(
    type(None), tp_int,
    lambda t,v: llvm.ee.GenericValue.int(t, 0),
    lambda t,v: llvm.core.Constant.int(t, 0)
)

_pyscalars = {
    int: Int,
    float: Float,
    bool: Bool
}
def get_scalar(obj):
    """obj can either be a value, or a type"""
    type_ = type(obj)
    if type_ == type(int):
        type_ = obj

    # HACK
    if type_ == type(None):
        return None_

    try:
        return _pyscalars[type_]
    except KeyError:
        raise TypingError("Invalid scalar type `{0}'".format(type_))

class ArrayType(Type):
    tp = NoType
    shape = None

    @classmethod
    def fromArray(klass, array):
        # TODO support more types
        if array.dtype == np.int64:
            dtype = _pyscalars[int]
        else:
            raise UnimplementedError("Numpy array dtype {0} not (yet) supported".format(array.dtype))

        # TODO: multidimensional arrays
        shape = array.shape[0]

        return ArrayType(dtype, shape)

    def __init__(self, tp, shape):
        assert tp in _pyscalars.values()
        self.tp = tp
        self.shape = shape
    def getElementType(self):
        return self.tp
    def llvmType(self):
        tp = llvm.core.Type.array(self.tp.llvmType(), self.shape)
        return llvm.core.Type.pointer(tp)
    def __str__(self):
        return "<{0}*{1}>".format(self.tp, self.shape)
    def __repr__(self):
        return str(self)

def supported_scalar(type_):
    if type(type_) == str:
        types = map(lambda x: x.__name__, _pyscalars.keys())
    else:
        types = _pyscalars.keys()
    return any([type_ == t for t in types])

def llvm_to_py(tp, val):
    if tp == Int:
        return val.as_int_signed()
    elif tp == Float:
        return val.as_real(py_type_to_llvm(tp))
    elif tp == Bool:
        return bool(val.as_int())
    elif tp == None_:
        return None
    else:
        raise TypingError("Unknown type {0}".format(tp))

def get(obj):
    """Resolve python object -> Stella type"""
    type_ = type(obj)
    if supported_scalar(type_):
        return get_scalar(type_)
    elif type_ == np.ndarray:
        return ArrayType.fromArray(obj)
    else:
        raise UnimplementedError("Unknown type {0}".format(type_))

#def llvm_constant(val):
#    tp = type(val)
#    elif tp == str:
#        return llvm.core.Constant.string(val)
#    # HACK {
#    elif tp == None.__class__:
#        return llvm.core.Constant.int(tp_int, 0)
#    # } HACK
#    else:
#        raise UnimplementedError("Unknown constant type {0}".format(tp))

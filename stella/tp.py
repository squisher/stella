import llvm
import llvm.core
import llvm.ee
import numpy as np
import ctypes

from . import exc


class Type(object):
    type_ = None
    _llvm = None
    ptr = 0

    def makePointer(self):
        """Note: each subtype must interpret `ptr` itself"""
        self.ptr += 1

    def isPointer(self):
        return self.ptr > 0

    def __str__(self):
        return '?'
    # llvm.core.ArrayType does something funny, it will compare against _ptr,
    # so let's just add the attribute here to enable equality tests
    _ptr = None

    def llvmType(self):
        raise exc.TypingError(
            "Cannot create llvm type for an unknown type. This should have been cought earlier.")

NoType = Type()


class ScalarType(Type):
    def __init__(self, name, type_, llvm, f_generic_value, f_constant):
        self.name = name
        self.type_ = type_
        self._llvm = llvm
        self.f_generic_value = f_generic_value
        self.f_constant = f_constant

    def llvmType(self):
        return self._llvm

    def genericValue(self, value):
        return self.f_generic_value(self._llvm, value)

    def constant(self, value, builder = None):
        return self.f_constant(self._llvm, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{0}:{1}>".format(str(type(self))[8:-2], self.name)

tp_int = llvm.core.Type.int(64)
tp_int32 = llvm.core.Type.int(32)  # needed for llvm operators
# tp_float = llvm.core.Type.float() # Python always works with double precision
tp_double = llvm.core.Type.double()
tp_bool = llvm.core.Type.int(1)
tp_void = llvm.core.Type.void()

Int = ScalarType(
    "Int",
    int, tp_int,
    llvm.ee.GenericValue.int_signed,
    llvm.core.Constant.int
)
uInt = ScalarType(  # TODO: unclear whether this is correct or not
    "uInt",
    int, tp_int32,
    llvm.ee.GenericValue.int,
    llvm.core.Constant.int
)
Float = ScalarType(
    "Float",
    float, tp_double,
    llvm.ee.GenericValue.real,
    llvm.core.Constant.real
)
Bool = ScalarType(
    "Bool",
    bool, tp_bool,
    llvm.ee.GenericValue.int,
    llvm.core.Constant.int
)


def invalid_none_use(msg):
    raise exc.StellaException(msg)
None_ = ScalarType(
    "NONE",
    type(None), tp_void,
    lambda t, v: invalid_none_use("Can't create a generic value ({0},{1}) for void".format(t, v)),
    lambda t, v: None  # Constant, needed for constructing `RETURN None'
)
Void = None_  # TODO: Could there be differences later?
Str = ScalarType(
    "Str",
    str, None,
    lambda t, v: None,
    lambda t, v: None
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

    # HACK {
    if type_ == type(None):  # noqa
        return None_
    elif type_ == str:
        return Str
    # } HACK

    try:
        return _pyscalars[type_]
    except KeyError:
        raise exc.TypingError("Invalid scalar type `{0}'".format(type_))


def supported_scalar(type_):
    """type_ is either a Python type or a stella type!"""
    try:
        get_scalar(type_)
        return True
    except exc.TypingError:
        # now check for stella scalar types
        return type_ in _pyscalars.values()


def supported_scalar_name(name):
    assert type(name) == str

    types = map(lambda x: x.__name__, _pyscalars.keys())
    return any([name == t for t in types])


class StructType(Type):
    attrib_type = None
    attrib_idx = None

    @classmethod
    def fromObj(klass, obj):
        attrib_type = {}
        attrib_idx = {}
        attrib_names = filter(lambda s: not s.startswith('_'), dir(obj))  # TODO: only exclude __?
        i = 0
        for name in attrib_names:
            attrib = getattr(obj, name)
            # TODO: catch the exception and improve the error message?
            type_ = get_scalar (type(attrib))
            attrib_type[name] = type_
            attrib_idx[name] = i
            i += 1

        # TODO Memoization of the results?
        return StructType(type(obj), attrib_names, attrib_type, attrib_idx)

    def __init__(self, type_, attrib_names, attrib_type, attrib_idx):
        self.name = str(type_)[1:-1]
        self.attrib_names = attrib_names
        self.attrib_type = attrib_type
        self.attrib_idx = attrib_idx

    def getMemberType(self, name):
        return self.attrib_type[name]

    def getMemberIdx(self, name):
        return self.attrib_idx[name]

    def baseType(self):
        return llvm.core.Type.struct([type_.llvmType() for type_ in self.attrib_type.values()])

    def llvmType(self):
        type_ = llvm.core.Type.pointer(self.baseType())
        if self.ptr:
            raise exc.UnimplementedError("Pointer to structs not allowed")
        return type_

    def constant(self, value, builder):
        type_ = self.baseType()
        result_llvm = builder.alloca(type_)
        # TODO free the memory!!! XXX
        for name in self.attrib_names:
            # insert_element(self, vec_val, elt_val, idx_val, name='')¶
            builder.insert_element(
                result_llvm,
                self.getMemberType().llvm,
                ir.Constant(self.attrib_idx[name]))
        return result_llvm


    def __str__(self):
        return "{}: {}".format(self.name, list(self.attrib_type.keys()))

    def __repr__(self):
        return "<{}>".format(self)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.attrib_names == other.attrib_names
                and self.attrib_types == other.attrib_types)

    def __ne__(self, other):
        return not self.__eq__(other)


class ArrayType(Type):
    tp = NoType
    shape = None

    @classmethod
    def fromArray(klass, array):
        # TODO support more types
        if array.dtype == np.int64:
            dtype = _pyscalars[int]
        else:
            raise exc.UnimplementedError("Numpy array dtype {0} not (yet) supported".format(
                array.dtype))

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
        type_ = llvm.core.Type.array(self.tp.llvmType(), self.shape)
        if self.ptr:
            type_ = llvm.core.Type.pointer(type_)
        return type_

    def __str__(self):
        if self.ptr:
            p = '*'
        else:
            p = ''
        return "<{0}{1}*{2}>".format(p, self.tp, self.shape)

    def __repr__(self):
        return str(self)


def llvm_to_py(tp, val):
    if tp == Int:
        return val.as_int_signed()
    elif tp == Float:
        return val.as_real(tp.llvmType())
    elif tp == Bool:
        return bool(val.as_int())
    elif tp is None_:
        return None
    else:
        raise exc.TypingError("Unknown type {0}".format(tp))


def get(obj):
    """Resolve python object -> Stella type"""
    type_ = type(obj)
    if supported_scalar(type_):
        return get_scalar(type_)
    elif type_ == np.ndarray:
        return ArrayType.fromArray(obj)
    else:
        return StructType.fromObj(obj)
        # TODO: How to identify unspported objects?
        #raise exc.UnimplementedError("Unknown type {0}".format(type_))

_cscalars = {
    ctypes.c_double: Float,
    ctypes.c_uint: uInt,
    None: Void
}


def from_ctype(type_):
    assert type(type_) == type(ctypes.c_int) or type(type_) == type(None)  # noqa
    return _cscalars[type_]

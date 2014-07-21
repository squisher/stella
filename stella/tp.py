import llvm
import llvm.core
import llvm.ee
import numpy as np
import ctypes
import logging

from . import exc


class Type(object):
    type_ = None
    _llvm = None
    ptr = 0
    on_heap = False

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

    def constant(self, value, module = None, builder = None):
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

def getIndex(i):
    if type(i) == int:
        return llvm.core.Constant.int(tp_int32, i)
    else:
        raise UnimplementedError("Unsupported index type {}".format(type(i)))


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
    base_type = None
    on_heap = True

    _singletons = dict()

    @classmethod
    def fromObj(klass, obj):
        type_name = str(type(obj))[1:-1]
        if type_name in klass._singletons:
            return klass._singletons[type_name]

        attrib_type = {}
        attrib_idx = {}
        attrib_names = list(filter(lambda s: not s.startswith('_'), dir(obj)))  # TODO: only exclude __?
        i = 0
        for name in attrib_names:
            attrib = getattr(obj, name)
            # TODO: catch the exception and improve the error message?
            type_ = get_scalar (type(attrib))
            attrib_type[name] = type_
            attrib_idx[name] = i
            i += 1

        type_ = StructType(type_name, attrib_names, attrib_type, attrib_idx)
        klass._singletons[type_name] = type_
        return type_

    def __init__(self, name, attrib_names, attrib_type, attrib_idx):
        self.name = name
        self.attrib_names = attrib_names
        self.attrib_type = attrib_type
        self.attrib_idx = attrib_idx

    def getMemberType(self, name):
        return self.attrib_type[name]

    def getMemberIdx(self, name):
        return self.attrib_idx[name]

    def baseType(self):
        if not self.base_type:
            llvm_types = [type_.llvmType() for type_ in self.attrib_type.values()]
            self.base_type = llvm.core.Type.struct(llvm_types, name=self.name)
        return self.base_type

    def llvmType(self):
        type_ = llvm.core.Type.pointer(self.baseType())
        if self.ptr > 1:
            raise exc.UnimplementedError("Pointer to (pointer of) structs not allowed")
        return type_

    def constant(self, value, module, builder):
        type_ = self.baseType()
        result_llvm = builder.alloca(type_)
        for name in self.attrib_names:
            idx_llvm = getIndex(self.attrib_idx[name])
            wrapped = wrapValue(getattr(value, name))
            wrapped_llvm = wrapped.translate(module, builder)
            p = builder.gep(result_llvm, [getIndex(0), idx_llvm], inbounds=True)
            builder.store(wrapped_llvm, p)

        return result_llvm


    def __str__(self):
        return "{}{}: {}".format('*'*self.ptr, self.name, list(self.attrib_type.keys()))

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
    on_heap = True

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


class Typable(object):
    type = NoType
    llvm = None

    def unify_type(self, tp2, debuginfo):
        tp1 = self.type
        if tp1 == tp2:
            pass
        elif tp1 == NoType:
            self.type = tp2
        elif tp2 == NoType:
            pass
        elif (tp1 == Int and tp2 == Float) or (tp1 == Float and tp2 == Int):
            self.type = Float
            return True
        else:
            raise exc.TypingError("Unifying of types {} and {} (not yet) implemented".format(
                tp1, tp2), debuginfo)

        return False

    def llvmType(self):
        """Map from Python types to LLVM types."""
        return self.type.llvmType()

    def translate(self, module, builder):
        return self.llvm


class Const(Typable):
    value = None

    def __init__(self, value):
        self.value = value
        try:
            self.type = get_scalar(value)
            self.name = str(value)
        except exc.TypingError as e:
            self.name = "InvalidConst({0}, type={1})".format(value, type(value))
            raise e


    def unify_type(self, tp2, debuginfo):
        r = super().unify_type(tp2, debuginfo)
        return r

    def translate(self, module, builder):
        self.llvm = self.type.constant(self.value, module, builder)
        return self.llvm

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class NumpyArray(Typable):
    def __init__(self, array):
        assert isinstance(array, np.ndarray)

        # TODO: multi-dimensional arrays
        self.type = ArrayType.fromArray(array)
        self.value = array

        ptr_int = self.value.ctypes.data  # int
        ptr_int_llvm = Int.constant(ptr_int)
        type_ = llvm.core.Type.pointer(self.type.llvmType())
        self.llvm = llvm.core.Constant.inttoptr(ptr_int_llvm, type_)

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return str(self)


class Struct(Const):
    def __init__(self, obj):
        self.type = StructType.fromObj(obj)
        self.value = obj

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return str(self)


def wrapValue(value):
    type_ = type(value)
    if supported_scalar(type_):
        return Const(value)
    elif type_ == np.ndarray:
        return NumpyArray(value)
    else:
        return Struct(value)


class Cast(Typable):
    def __init__(self, obj, tp):
        assert obj.type != tp
        self.obj = obj
        self.type = tp
        self.emitted = False

        logging.debug("Casting {0} to {1}".format(self.obj.name, self.type))
        self.name = "({0}){1}".format(self.type, self.obj.name)

    def translate(self, module, builder):
        if self.emitted:
            assert hasattr(self, 'llvm')
            return self.llvm
        self.emitted = True

        # TODO: HACK: instead of failing, let's make it a noop
        # assert self.obj.type == int and self.type == float
        if self.obj.type == self.type:
            self.llvm = self.obj.llvm
            return self.llvm

        if isinstance(self.obj, Const):
            value = float(self.obj.value)
            self.llvm = self.obj.type.constant(value)
        else:
            self.llvm = builder.sitofp(self.obj.llvm, Float.llvmType(), self.name)
        return self.llvm

    def __str__(self):
        return self.name

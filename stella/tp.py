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
        raise exc.TypeError(
            "Cannot create llvm type for an unknown type. This should have been cought earlier.")


NoType = Type()


class ScalarType(Type):
    def __init__(self, name, type_, llvm, ctype, f_generic_value, f_constant):
        self.name = name
        self.type_ = type_
        self.ctype = ctype
        self._llvm = llvm
        self.f_generic_value = f_generic_value
        self.f_constant = f_constant

    def llvmType(self):
        return self._llvm

    def genericValue(self, value):
        return self.f_generic_value(self._llvm, value)

    def constant(self, value, cge = None):
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
    int, tp_int, ctypes.c_int64,
    llvm.ee.GenericValue.int_signed,
    llvm.core.Constant.int
)
uInt = ScalarType(  # TODO: unclear whether this is correct or not
    "uInt",
    int, tp_int32, ctypes.c_int32,
    llvm.ee.GenericValue.int,
    llvm.core.Constant.int
)
Float = ScalarType(
    "Float",
    float, tp_double, ctypes.c_double,
    llvm.ee.GenericValue.real,
    llvm.core.Constant.real
)
Bool = ScalarType(
    "Bool",
    bool, tp_bool, ctypes.c_bool,
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
    type(None), tp_void, ctypes.c_void_p,
    lambda t, v: invalid_none_use("Can't create a generic value ({0},{1}) for void".format(t, v)),
    lambda t, v: None  # Constant, needed for constructing `RETURN None'
)
Void = None_  # TODO: Could there be differences later?
Str = ScalarType(
    "Str",
    str, None, ctypes.c_char_p,
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
        raise exc.TypeError("Invalid scalar type `{0}'".format(type_))


def supported_scalar(type_):
    """type_ is either a Python type or a stella type!"""
    try:
        get_scalar(type_)
        return True
    except exc.TypeError:
        # now check for stella scalar types
        return type_ in _pyscalars.values()


def supported_scalar_name(name):
    assert type(name) == str

    types = map(lambda x: x.__name__, _pyscalars.keys())
    return any([name == t for t in types])


class StructType(Type):

    on_heap = True

    attrib_type = None
    attrib_idx = None
    base_type = None
    type_store = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        type_name = str(type(obj))[1:-1]
        attrib_type = {}
        attrib_idx = {}
        attrib_names = list(filter(lambda s: not s.startswith('_'), dir(obj)))  # TODO: only exclude __?
        i = 0
        for name in attrib_names:
            attrib = getattr(obj, name)
            # TODO: catch the exception and improve the error message?
            type_ = get (attrib)
            attrib_type[name] = type_
            attrib_idx[name] = i
            i += 1

        type_ = StructType(type_name, attrib_names, attrib_type, attrib_idx)
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

    def llvmType(self):
        # TODO find a more efficient place where the pointer values are created
        # This needs to be done before the cache look up because it will
        # affect mangled_name.
        for type_ in self.attrib_type.values():
            if type_.on_heap:
                type_.makePointer()
        mangled_name = '_'.join([self.name] + [str(t) for t in self.attrib_type.values()])

        if not mangled_name in self.__class__.type_store:
            llvm_types = []
            for type_ in self.attrib_type.values():
                llvm_types.append(type_.llvmType())
            type_ = llvm.core.Type.struct(llvm_types, name=mangled_name)
            ptype_ = llvm.core.Type.pointer(type_)
            self.__class__.type_store[mangled_name] = ptype_
            return ptype_
        else:
            return self.__class__.type_store[mangled_name]

    def ctypes(self):
        fields = []
        for name in self.attrib_names:
            fields.append((name, self.attrib_type[name].ctype))
        ctype = type("_" + self.name + "_transfer", (ctypes.Structure, ), {'_fields_': fields})
        return ctype

    def constant(self, value, cge):
        """Transfer values Python -> Stella"""
        ctype = self.ctypes()
        transfer_value = ctype()

        for name in self.attrib_names:
            item = getattr(value, name)
            if isinstance(item, np.ndarray):
                item = ctypes.cast(item.ctypes.data, ctypes.POINTER(ctypes.c_int))
            setattr(transfer_value, name, item)

        addr_llvm = Int.constant(int(ctypes.addressof(transfer_value)))
        result_llvm = cge.builder.inttoptr(addr_llvm,
                                           self.llvmType())
        return (result_llvm, transfer_value)


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
    ctype = ctypes.POINTER(ctypes.c_int)  # TODO why is ndarray.ctypes.data of type int?

    @classmethod
    def fromArray(klass, array):
        # TODO support more types
        if array.dtype == np.int64:
            dtype = _pyscalars[int]
        elif array.dtype == np.float64:
            dtype = _pyscalars[float]
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
        raise exc.TypeError("Unknown type {0}".format(tp))


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
            raise exc.TypeError("Unifying of types {} and {} (not yet) implemented".format(
                tp1, tp2), debuginfo)

        return False

    def llvmType(self):
        """Map from Python types to LLVM types."""
        return self.type.llvmType()

    def translate(self, cge):
        return self.llvm

    def copy2Python(self, cge):
        pass


class Const(Typable):
    value = None

    def __init__(self, value):
        self.value = value
        try:
            self.type = get_scalar(value)
            self.name = str(value)
        except exc.TypeError as e:
            self.name = "InvalidConst({0}, type={1})".format(value, type(value))
            raise e


    def unify_type(self, tp2, debuginfo):
        r = super().unify_type(tp2, debuginfo)
        return r

    def translate(self, cge):
        if self.type.on_heap:
            (self.llvm, self.transfer_value) = self.type.constant(self.value, cge)
        else:
            self.llvm = self.type.constant(self.value, cge)
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

    def copy2Python(self, cge):
        for name in self.type.attrib_names:
            item = getattr(self.transfer_value, name)
            if not self.type.attrib_type[name].on_heap:
                setattr(self.value, name, item)
        del self.transfer_value


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

    def translate(self, cge):
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
            self.llvm = self.type.constant(value)
        else:
            self.llvm = cge.builder.sitofp(self.obj.llvm, Float.llvmType(), self.name)
        return self.llvm

    def __str__(self):
        return self.name

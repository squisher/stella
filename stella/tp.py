import llvm
import llvm.core
import llvm.ee
import numpy as np
import ctypes
import logging
import types
from abc import ABCMeta, abstractmethod
import inspect

from . import exc


class Type(object):
    type_ = None
    _llvm = None
    ptr = 0
    on_heap = False

    def makePointer(self):
        """Note: each subtype must interpret `ptr` itself"""
        # TODO: clean up calling of makePointer; it happes too frequently
        #self.ptr += 1
        if self.on_heap:
            self.ptr = 1

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


class PyWrapper(Type):
    """Wrap Python types, e.g. for the intrinsic zeros dtype parameter.

    This allows passing types as first-class values, but is used only in
    special circumstances like zeros().
    """
    def __init__(self, py):
        self.py = py
        self.bc = None
        self.type = get_scalar(py)

    def makePointer(self):
        raise exc.TypeError("Cannot make a pointer of Python type {}".format(self.py))

    def isPointer(self):
        raise exc.TypeError("Cannot check pointer status of Python type {}".format(self.py))

    def __str__(self):
        return str(self.py)

    def llvmType(self):
        raise exc.TypeError("Cannot create an LLVM type for Python type {}".format(self.py))


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
        raise exc.UnimplementedError("Unsupported index type {}".format(type(i)))


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
    elif type_ == PyWrapper:
        type_ = obj.py

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

    attrib_names = None
    attrib_type = None
    attrib_idx = None
    base_type = None
    type_store = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        type_name = str(type(obj)).split("'")[1]
        attrib_type = {}
        attrib_idx = {}
        attrib_names = sorted(list(filter(lambda s: not s.startswith('_'),
                                          dir(obj))))  # TODO: only exclude __?
        for name in attrib_names:
            attrib = getattr(obj, name)
            # TODO: catch the exception and improve the error message?
            type_ = get (attrib)
            attrib_type[name] = type_

        # Sort attrib_names so that function types are after attribute types.
        # This allows me to keep them around, because even though they aren't
        # translated into the llvm struct, their presence does not mess up the
        # indices.
        def funcs_last(n):
            if isinstance(attrib_type[n], FunctionType):
                return 1
            else:
                return 0

        attrib_names = sorted(attrib_names, key=funcs_last)
        i = 0
        for name in attrib_names:
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

    def _scalarAttributeNames(self):
        return filter(lambda n: not isinstance(self.attrib_type[n], FunctionType),
                      self.attrib_names)

    def llvmType(self):
        # TODO this was here for mangled_name, is it still necessary?
        for type_ in self.attrib_type.values():
            if type_.on_heap:
                type_.makePointer()
        # Name is fully qualified and hence unique, no need to mangle a name
        # anymore.
        mangled_name = self.name

        if mangled_name not in self.__class__.type_store:
            llvm_types = []
            for name in self._scalarAttributeNames():
                type_ = self.attrib_type[name]
                llvm_types.append(type_.llvmType())
            type_ = llvm.core.Type.struct(llvm_types, name=mangled_name)
            ptype_ = llvm.core.Type.pointer(type_)
            self.__class__.type_store[mangled_name] = ptype_
            return ptype_
        else:
            return self.__class__.type_store[mangled_name]

    def ctypes(self):
        fields = []
        for name in self._scalarAttributeNames():
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
        return "{}{}".format('*'*self.ptr, self.name)

    def __repr__(self):
        #return "<{}>".format(self)
        return "<{}{}: {}>".format('*'*self.ptr, self.name, list(self.attrib_type.keys()))

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.attrib_names == other.attrib_names
                and self.attrib_type == other.attrib_type)

    def __ne__(self, other):
        return not self.__eq__(other)


class ArrayType(Type):
    type_ = NoType
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

    def __init__(self, type_, shape):
        assert type_ in _pyscalars.values()
        self.type_ = type_
        self.shape = shape

    def getElementType(self):
        return self.type_

    def llvmType(self):
        type_ = llvm.core.Type.array(self.type_.llvmType(), self.shape)
        if self.ptr:
            type_ = llvm.core.Type.pointer(type_)

        return type_

    def __str__(self):
        return "<{}{}*{}>".format('*'*self.ptr, self.type_, self.shape)

    def __repr__(self):
        return str(self)


class Callable(metaclass=ABCMeta):
    @abstractmethod
    def getResult(self, func):
        pass

    def combineArgs(self, args, kwargs):
        """Combine concrete args and kwargs according to calling conventions.

        Precondition: Typing has been performed, so typeArgs already ensures
        that the correct number of arguments are provided.
        """
        return self.type_._combineArgs(args, kwargs)

    def call(self, cge, args, kw_args):
        combined_args = self.combineArgs(args, kw_args)

        return cge.builder.call(self.llvm, [arg.translate(cge) for arg in combined_args])


class Foreign(object):
    """Mixin: This is not a Python function. It does not need to get analyzed."""
    pass


class FunctionType(Type):
    _registry = {}

    @classmethod
    def get(klass, obj, bound=None, builtin=False):
        if bound:
            key = (type(bound), obj.__name__)
        else:
            key = obj

        if key not in klass._registry:
            klass._registry[key] = klass(obj, bound, builtin)

        return klass._registry[key]

    @classmethod
    def destruct(klass):
        klass._registry.clear()

    def __init__(self, obj, bound=None, builtin=False):
        """Type representing a function.

        obj: Python function reference
        bound: self if it is a method
        builtin: True if e.g. len

        Assumption: bound or builtin
        """
        self.name = obj.__name__
        self._func = obj
        self.bound = bound
        self._builtin = builtin

        self.readSignature(obj)

    def pyFunc(self):
        return self._func

    @property
    def bound(self):
        """None if a regular function, returns the type of self if a bound method

        Note that unbound methods are not yet supported
        """
        # Lazily get the type of bound to avoid recursion:
        # -> self is a struct, which has as one of its members a bound function
        if self._bound is None:
            return None
        return get(self._bound)

    @bound.setter
    def bound(self, obj):
        self._bound = obj

    @property
    def builtin(self):
        return self._builtin

    arg_defaults = []
    tp_defaults = []
    arg_names = []
    arg_types = []
    def_offset = 0

    @abstractmethod
    def getReturnType(self, args, kw_args):
        pass

    def readSignature(self, f):
        argspec = inspect.getargspec(f)
        self.arg_names = argspec.args
        self.arg_defaults = [Const(default) for default in argspec.defaults or []]
        self.tp_defaults = [d.type for d in self.arg_defaults]
        self.def_offset = len(self.arg_names)-len(self.arg_defaults)

    def typeArgs(self, tp_args, tp_kwargs):
        # TODO store the result?

        if self.bound:
            tp_args.insert(0, self.bound)

        num_args = len(tp_args)
        if num_args+len(tp_kwargs) < len(self.arg_names)-len(self.arg_defaults):
            raise exc.TypeError("takes at least {0} argument(s) ({1} given)".format(
                len(self.arg_names)-len(self.arg_defaults), len(tp_args)+len(tp_kwargs)))
        if num_args+len(tp_kwargs) > len(self.arg_names):
            raise exc.TypeError("takes at most {0} argument(s) ({1} given)".format(
                len(self.arg_names), len(tp_args)))

        if len(self.arg_types) == 0:
            self.arg_types = self._combineArgs(tp_args, tp_kwargs, self.tp_defaults)
        else:
            # Already typed, so the supplied arguments must match what the last
            # call used.
            supplied_args = self._combineArgs(tp_args, tp_kwargs, self.tp_defaults)
            for i, prototype, supplied in zip(range(len(supplied_args)),
                                              self.arg_types,
                                              supplied_args):
                if prototype != supplied:
                    raise exc.TypeError("Argument {} has type {}, but type {} was supplied".format(
                        self.arg_names[i], prototype, supplied))
        return self.arg_types

    def _combineArgs(self, args, kwargs, defaults=None):
        """Combine concrete or types of args and kwargs according to calling conventions."""
        if defaults is None:
            defaults = self.arg_defaults
        num_args = len(args)
        r = [None] * len(self.arg_names)

        # copy supplied regular arguments
        for i in range(len(args)):
            r[i] = args[i]

        # set default values
        for i in range(max(num_args, len(self.arg_names)-len(defaults)),
                       len(self.arg_names)):
            r[i] = defaults[i-self.def_offset]

        # insert kwargs
        for k, v in kwargs.items():
            try:
                idx = self.arg_names.index(k)
                if idx < num_args:
                    raise exc.TypeError("got multiple values for keyword argument '{0}'".format(
                        self.arg_names[idx]))
                r[idx] = v
            except ValueError:
                raise exc.TypeError("Function does not take an {0} argument".format(k))

        return r

    def __str__(self):
        if self._bound:
            tp_name = str(type(self._bound)).split("'")[1]
            return "<bound method {}.{}>".format(tp_name, self._func.__name__)
        else:
            return "<function {}>".format(self._func.__name__)

    @property
    def fq(self):
        """Returns the fully qualified type name."""
        if self._bound:
            return "{}.{}".format(str(self.bound), self.name)
        else:
            return self.name

    def llvmType(self):
        raise exc.InternalError("This is an intermediate type presentation only!")


class IntrinsicType(FunctionType):
    def __init__(self, f, names, defaults):
        super().__init__(f, bound=None, builtin=True)
        self.arg_names = names
        self.arg_defaults = defaults
        self.def_offset = len(self.arg_names)-len(self.arg_defaults)

    def readSignature(self, f):
        """The signature is built in for Intrinsics. NOOP."""
        pass


class ExtFunctionType(Foreign, FunctionType):
    def __init__(self, signature):
        ret, arg_types = signature
        self.return_type = from_ctype(ret)
        self.arg_types = list(map(from_ctype, arg_types))
        self.readSignature(None)

    def __str__(self):
        return "<{} function({})>".format(self.return_type,
                                           ", ".join(zip(self.arg_types, self.arg_names)))

    def readSignature(self, f):
        # arg, inspect.getargspec(f) doesn't work for C/cython functions
        self.arg_names = ['arg{0}' for i in range(len(self.arg_types))]
        self.arg_defaults = []

    def getReturnType(self, args, kw_args):
        return self.return_type


def llvm_to_py(type_, val):
    if type_ == Int:
        return val.as_int_signed()
    elif type_ == Float:
        return val.as_real(type_.llvmType())
    elif type_ == Bool:
        return bool(val.as_int())
    elif type_ is None_:
        return None
    else:
        raise exc.TypeError("Unknown type {0}".format(type_))


def get(obj):
    """Resolve python object -> Stella type"""
    type_ = type(obj)
    if supported_scalar(type_):
        return get_scalar(type_)
    elif type_ == np.ndarray:
        return ArrayType.fromArray(obj)
    elif isinstance(obj, types.FunctionType):
        return FunctionType.get(obj)
    elif isinstance(obj, types.MethodType):
        return FunctionType.get(obj, bound=obj.__self__)
    elif isinstance(obj, types.BuiltinFunctionType):
        return FunctionType.get(obj, builtin=True)
    elif isinstance(obj, types.BuiltinMethodType):
        assert False and "TODO: This case has not been completely implemented"
        return FunctionType.get(obj, bound=True, builtin=True)
    else:
        # TODO: How to identify unspported objects? Everything is an object...
        #raise exc.UnimplementedError("Unknown type {0}".format(type_))
        return StructType.fromObj(obj)

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
        """Returns: widened:bool, needs_cast:bool
        widened: this type changed
        needs_cast: tp2 needs to be cast to this type
        """
        tp1 = self.type
        if tp1 == tp2:
            pass
        elif tp1 == NoType:
            self.type = tp2
        elif tp2 == NoType:
            pass
        elif tp1 == Int and tp2 == Float:
            self.type = Float
            return True, False
        elif tp1 == Float and tp2 == Int:
            # Note that the type does not have to change here because Float is
            # already wider than Int
            return False, True
        else:
            raise exc.TypeError("Unifying of types {} and {} (not yet) implemented".format(
                tp1, tp2), debuginfo)

        return False, False

    def llvmType(self):
        """Map from Python types to LLVM types."""
        return self.type.llvmType()

    def translate(self, cge):
        return self.llvm

    def copy2Python(self, cge):
        pass


class ImmutableType(object):
    def unify_type(self, tp2, debuginfo):
        raise TypeError("Type {} is immutable, it cannot be unified with {} at {}".format(
            self.type, tp2, debuginfo))

    def llvmType(self):
        """Map from Python types to LLVM types."""
        return self.type.llvmType()


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


class Struct(Typable):
    obj_store = {}  # Class variable

    @classmethod
    def new(klass, obj):
        """Only one Struct representation per Python object instance.
        """
        if not hasattr(obj, '__stella_wrapper__'):
            obj.__stella_wrapper__ = Struct(obj)
        assert isinstance(obj.__stella_wrapper__, klass)
        return obj.__stella_wrapper__

    def __init__(self, obj):
        self.type = StructType.fromObj(obj)
        self.value = obj

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return repr(self.type)

    def translate(self, cge):
        if not self.llvm:
            (self.llvm, self.transfer_value) = self.type.constant(self.value, cge)
        return self.llvm
    def copy2Python(self, cge):
        """At the end of a Stella run, the struct's values need to be copied back
        into Python.
        """
        for name in self.type.attrib_names:
            item = getattr(self.transfer_value, name)
            if not self.type.attrib_type[name].on_heap:
                setattr(self.value, name, item)
        del self.transfer_value
        del self.value.__stella_wrapper__


def wrapValue(value):
    type_ = type(value)
    if supported_scalar(type_):
        return Const(value)
    elif type_ == np.ndarray:
        return NumpyArray(value)
    else:
        return Struct.new(value)


class Cast(Typable):
    def __init__(self, obj, tp):
        assert obj.type != tp
        self.obj = obj
        self.type = tp
        self.emitted = False

        logging.debug("Casting {0} to {1}".format(self.obj.name, self.type))
        self.name = "({0}){1}".format(self.type, self.obj.name)

    def translate(self, cge):
        """This is a special case:
        The .llvm attribute is set by bytecode that is being cast here.
        So save it in obj, and generate our own .llvm
        """
        if self.emitted:
            return self.llvm
        self.emitted = True

        # TODO: HACK: instead of failing, let's make it a noop
        # I need to check WHY these casts are being created and if I can avoid
        # them
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


def destruct():
    FunctionType.destruct()
    StructType.type_store.clear()

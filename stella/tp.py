import llvmlite.ir as ll
import numpy as np
import ctypes
import logging
import types
from abc import ABCMeta, abstractmethod
import inspect

from . import exc


class Type(metaclass=ABCMeta):
    type_ = None
    _llvm = None
    ptr = 0
    on_heap = False
    req_transfer = False
    ctype = False  # init to false because None is a valid ctype

    def makePointer(self, ensure = False):
        if self.on_heap:
            if ensure:
                self.ptr = 1
            else:
                assert self.ptr == 0
                self.ptr += 1

    def isReference(self):
        return self.ptr > 0

    def isUntyped(self):
        return False

    def dereference(self):
        if self.ptr == 1:
            # TODO hackish!
            return self
        else:
            raise exc.TypeError("Cannot dereference the non-reference type {}".format(self))

    @abstractmethod
    def __str__(self):
        pass

    def llvmType(self, module):
        # some types just come as a reference (e.g. external numpy array). For
        # references within stella use class Reference.
        assert self.ptr <= 1
        type_ = self._llvmType(module)
        if self.ptr:
            return type_.as_pointer()
        else:
            return type_

    def _llvmType(self, module):
        raise exc.TypeError(
            "Cannot create llvm type for an unknown type. This should have been cought earlier.")

    def Ctype(self):
        if type(self.ctype) == bool:
            raise exc.TypeError(
                "Cannot create ctype for an unknown type. This should have been cought earlier.")
        else:
            return self.ctype

    def getElementType(self, idx):
        return NoType


class UnknownType(Type):
    def isUntyped(self):
        return False

    def __str__(self):
        return '?'

    def dereference(self):
        return self


NoType = UnknownType()


class Reference(Type):
    def __init__(self, type_):
        self.type_ = type_
        self.ptr = type_.ptr + 1

    def llvmType(self, module):
        type_ = self.type_.llvmType(module)
        # for i in range(self.ptr):
        return type_.as_pointer()

    def dereference(self):
        return self.type_

    def __str__(self):
        return '*{}'.format(self.type_)

class Subscriptable(metaclass=ABCMeta):
    """Mixin"""
    @abstractmethod
    def loadSubscript(cge, container, idx):
        pass

    @abstractmethod
    def storeSubscript(cge, container, idx, value):
        pass


class PyWrapper(Type):
    """Wrap Python types, e.g. for the intrinsic zeros dtype parameter.

    This allows passing types as first-class values, but is used only in
    special circumstances like zeros().
    """
    def __init__(self, py):
        self.py = py
        self.bc = None
        self.type = get_scalar(py)

    #def makePointer(self):
    #    raise exc.TypeError("Cannot make a pointer of Python type {}".format(self.py))

    #def isPointer(self):
    #    raise exc.TypeError("Cannot check pointer status of Python type {}".format(self.py))

    def __str__(self):
        return str(self.py)

    def _llvmType(self, module):
        raise exc.TypeError("Cannot create an LLVM type for Python type {}".format(self.py))


class ScalarType(Type):
    def __init__(self, name, type_, llvm, ctype):
        self.name = name
        self.type_ = type_
        self.ctype = ctype
        self._llvm = llvm

    def llvmType(self, module = None):
        return super().llvmType(module)

    def _llvmType(self, module):
        return self._llvm

    def constant(self, value, cge = None):
        return ll.Constant(self._llvm, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{0}:{1}>".format(str(type(self))[8:-2], self.name)


tp_int = ll.IntType(64)
tp_int32 = ll.IntType(32)  # needed for llvm operators
tp_double = ll.DoubleType()
tp_bool = ll.IntType(1)
tp_void = ll.VoidType()

Int = ScalarType(
    "Int",
    int, tp_int, ctypes.c_int64,
)
uInt = ScalarType(  # TODO: unclear whether this is correct or not
    "uInt",
    int, tp_int32, ctypes.c_int32,
)
Float = ScalarType(
    "Float",
    float, tp_double, ctypes.c_double,
)
Bool = ScalarType(
    "Bool",
    bool, tp_bool, ctypes.c_bool,
)

def getIndex(i):
    if type(i) == int:
        return ll.Constant(tp_int32, i)
    else:
        raise exc.UnimplementedError("Unsupported index type {}".format(type(i)))


def invalid_none_use(msg):
    raise exc.StellaException(msg)


None_ = ScalarType(
    "NONE",
    type(None), tp_void, None,
)
Void = None_  # TODO: Could there be differences later?
Str = ScalarType(
    "Str",
    str, None, ctypes.c_char_p,
)

_pyscalars = {
    int: Int,
    float: Float,
    bool: Bool,
}


def get_scalar(obj):
    """obj can either be a value, or a type

    Returns the Stella type for the given object"""
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


class CType(object):
    """
    Dynamically create a ctype Structure.
    """
    _registry = {}

    @classmethod
    def getStruct(klass, name, fields = []):
        """
        Creates a Structure with the given fields. Caches based on (name, fields).
        """
        fields = tuple(fields)
        if name not in klass._registry:
            if fields:
                attribs = {'_fields_': fields}
            else:
                attribs = {}
            type_ = type(name, (ctypes.Structure, ), attribs)
            klass._registry[name] = type_
            return type_
        else:
            struct = klass._registry[name]
            if fields:
                assert fields == struct._fields_
            return struct

    @classmethod
    def destruct(klass):
        klass._registry.clear()


class StructType(Type):
    on_heap = True
    req_transfer = True

    attrib_names = None
    attrib_type = None
    attrib_idx = None
    _ctype = None
    _llvmtype = None
    type_store = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        if type(obj) == type(int):
            # class
            type_name = str(obj).split("'")[1] + ".__class__"
        else:
            # instance
            type_name = str(type(obj)).split("'")[1]

        # cache it early, which allows fields of this type to be resolved
        # immediately
        if type_name in klass.type_store:
            return klass.type_store[type_name]

        type_ = StructType(type_name)
        type_.makePointer()  # by default
        klass.type_store[type_name] = type_

        attrib_type = {}
        attrib_idx = {}
        attrib_names = sorted(list(filter(lambda s: not s.startswith('_'),
                                          dir(obj))))  # TODO: only exclude __?
        for name in attrib_names:
            attrib = getattr(obj, name)
            try:
                a_type = get (attrib)
            except TypeError:
                raise exc.TypeError("{}({}).{}({}) is not supported".format(obj, type(obj), name, type(attrib)))
            if a_type.on_heap:
                #a_type = Reference(type_)
                a_type.makePointer(True)
            attrib_type[name] = a_type

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

        type_.attrib_type = attrib_type
        type_.attrib_idx = attrib_idx
        type_.attrib_names = attrib_names

        return type_

    @classmethod
    def destruct(klass):
        klass.type_store.clear()

    def __init__(self, name):
        self.name = name

    def getMemberType(self, name):
        return self.attrib_type[name]

    def getMemberIdx(self, name):
        return self.attrib_idx[name]

    def _scalarAttributeNames(self):
        return filter(lambda n: not isinstance(self.attrib_type[n], FunctionType),
                      self.attrib_names)

    def items(self):
        """Return (unordered) name, type tuples
        """
        # TODO turn this into an iterator?
        return [(name, self.attrib_type[name]) for name in self.attrib_names]
    def _llvmType(self, module):
        if self._llvmtype is not None:
            return self._llvmtype

        self._llvmtype = module.llvm.context.get_identified_type(self.name)
        assert self._llvmtype.is_opaque

        llvm_types = []
        for name in self._scalarAttributeNames():
            type_ = self.attrib_type[name]
            llvm_types.append(type_.llvmType(module))
        #type_ = ll.LiteralStructType(llvm_types)
        #self._llvmtype = type_
        self._llvmtype.set_body(*llvm_types)

        return self._llvmtype

    @property
    def ctype(self):
        if self._ctype:
            return self._ctype
        fields = []
        self._ctype = CType.getStruct("_" + self.name + "_transfer")
        for name in self._scalarAttributeNames():
            if isinstance(self.attrib_type[name], ListType):
                fields.append((name, ctypes.POINTER(self.attrib_type[name].ctype)))
            elif self.attrib_type[name] is self:
                fields.append((name, ctypes.POINTER(self._ctype)))
            else:
                fields.append((name, self.attrib_type[name].ctype))
        self._ctype._fields_ = fields
        return self._ctype

    def ctypeInit(self, value, transfer_value):
        for name in self.attrib_names:
            item = getattr(value, name)
            if isinstance(item, np.ndarray):
                # TODO: will this fail with float?
                item = ctypes.cast(item.ctypes.data, ctypes.POINTER(ctypes.c_int))
            elif isinstance(item, list):
                l = List.fromObj(item)
                l.ctypeInit()
                item = ctypes.cast(ctypes.addressof(l.transfer_value), ctypes.POINTER(l.type.ctype))
            elif self.attrib_type[name] is self:
                item = ctypes.cast(ctypes.addressof(transfer_value), ctypes.POINTER(self.ctype))
            elif not supported_scalar(type(item)):
                # struct, has to be the last check because everything is an
                # object in Python
                item = wrapValue(item).transfer_value
            setattr(transfer_value, name, item)

    def constant(self, value, cge):
        """Transfer values Python -> Stella"""
        transfer_value = self.ctype()

        assert self.ptr == 1

        self.ctypeInit(value, transfer_value)

        addr_llvm = Int.constant(int(ctypes.addressof(transfer_value)))
        result_llvm = ll.Constant(tp_int, addr_llvm).inttoptr(self.llvmType(cge.module))
        return (result_llvm, transfer_value)

    def ctype2Python(self, transfer_value, value):
        for name in self._scalarAttributeNames():
            item = getattr(transfer_value, name)
            # TODO generalize!
            if isinstance(self.attrib_type[name], List):
                l = List.fromObj(item)
                l.ctype2Python(item)
            elif not self.attrib_type[name].on_heap:
                # TODO is this actually used?
                setattr(value, name, item)

    def resetReference(self):
        """Special case: when a list of objects is allocated, then the type is NOT a pointer type"""
        self.ptr = 0

    def __str__(self):
        return "{}{}".format('*'*self.ptr, self.name)

    def __repr__(self):
        if self.attrib_type:
            type_info = list(self.attrib_type.keys())
        else:
            type_info = '?'
        return "<{}{}: {}>".format('*'*self.ptr, self.name, type_info)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.attrib_names == other.attrib_names
                and self.attrib_type == other.attrib_type)

    def __ne__(self, other):
        return not self.__eq__(other)


class TupleType(ScalarType, Subscriptable):
    def __init__(self, types):
        self.types = types

    def _getConst(self, idx):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, Const):
            return idx.value
        else:
            raise exc.TypeError("Tuple index must be constant, not {}".format(type(idx)))

    def getElementType(self, idx):
        val = self._getConst(idx)
        if val >= len(self.types):
            raise exc.IndexError("tuple index out of range")
        return self.types[val]

    def loadSubscript(self, cge, container, idx):
        val = self._getConst(idx)
        return cge.builder.extract_value(container.translate(cge), [val])

    def storeSubscript(self, cge, container, idx, value):
        # TODO Needs tests!
        idx_val = self._getConst(idx)
        cge.builder.insert_value(container.translate(cge), value.translate(cge), idx_val)

    def Ctype(self):
        fields = [("f{}".format(i), val.ctype) for i, val in enumerate(self.types)]
        return CType.getStruct("__tuple__", fields)

    @classmethod
    def unpack(klass, val):
        """
        Convert a ctypes.Structure wrapper into a native tuple.
        """
        l = [getattr(val, n) for n, _ in val._fields_]
        return tuple(l)

    def unify_type(self, values):
        for mv, ov in zip(self.values, values):
            mv.type.unify_type(ov.type)

    def _llvmType(self, module):
        return ll.LiteralStructType([t.llvmType(module) for t in self.types])

    def constant(self, values, cge = None):
        if not self._llvm:
            self._llvm = ll.Constant.literal_struct([wrapValue(v).translate(cge) for v in values])
        return self._llvm

    def __str__(self):
        return "tuple, {} elems".format(len(self.types))

    def __repr__(self):
        return "{}".format(", ".join([str(t) for t in self.types]))


class ArrayType(Type, Subscriptable):
    type_ = NoType
    shape = None
    on_heap = True
    ctype = ctypes.POINTER(ctypes.c_int)  # TODO why is ndarray.ctypes.data of type int?

    @classmethod
    def fromObj(klass, obj):
        # TODO support more types
        if obj.dtype == np.int64:
            dtype = _pyscalars[int]
        elif obj.dtype == np.float64:
            dtype = _pyscalars[float]
        else:
            raise exc.UnimplementedError("Numpy array dtype {0} not (yet) supported".format(
                obj.dtype))

        # TODO: multidimensional arrays
        shape = obj.shape[0]

        assert klass.isValidType(dtype)

        return ArrayType(dtype, shape)

    @classmethod
    def isValidType(klass, type_):
        return type_ in _pyscalars.values()

    def __init__(self, type_, shape):
        self.type_ = type_
        self.shape = shape

    def _boundsCheck(self, idx):
        """Check bounds, if possible. This is a compile time operation."""
        if isinstance(idx, Const) and idx.value >= self.shape:
            raise exc.IndexError("array index out of range")

    def getElementType(self, idx):
        self._boundsCheck(idx)
        return self.type_

    def _llvmType(self, module):
        type_ = ll.ArrayType(self.type_.llvmType(module), self.shape)

        return type_

    def __str__(self):
        return "{}{}[{}]".format('*'*self.ptr, self.type_, self.shape)

    def __repr__(self):
        return '<{}>'.format(self)

    def loadSubscript(self, cge, container, idx):
        self._boundsCheck(idx)
        p = cge.builder.gep(container.translate(cge),
                            [Int.constant(0), idx.translate(cge)],
                            inbounds=True)
        return cge.builder.load(p)

    def storeSubscript(self, cge, container, idx, value):
        self._boundsCheck(idx)
        p = cge.builder.gep(
            container.translate(cge), [
                Int.constant(0), idx.translate(cge)], inbounds=True)
        cge.builder.store(value.translate(cge), p)


class ListType(ArrayType):
    req_transfer = True
    type_store = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        # type checking: only continue if the list can be represented.
        if len(obj) == 0:
            raise exc.TypeError("Empty lists are not supported, because they are not typable.")
        type_ = type(obj[0])
        for o in obj[1:]:
            if type_ != type(o):
                raise exc.TypeError("List contains elements of type {} and type {}, but lists must not contain objects of more than one type.".format(type_, type(o)))

        base_type = get(obj[0])
        if not isinstance(base_type, StructType):
            msg = "Python lists must contain objects, not {}. Use numpy arrays for simple types.".format(base_type)
            raise exc.TypeError(msg)
        base_type.resetReference()
        # assert !klass.isValidType(dtype)

        # type_name = "[{}]".format(str(type(obj[0])).split("'")[1])
        type_ = klass(base_type, len(obj))
        return type_

    @classmethod
    def destruct(klass):
        klass.type_store.clear()

    def __init__(self, base_type, shape):
        super().__init__(base_type, shape)

    def _llvmType(self, module):
        mangled_name = str(self)

        if mangled_name not in self.__class__.type_store:
            type_ = ll.ArrayType(self.type_.llvmType(module), self.shape)
            self.__class__.type_store[mangled_name] = type_
            return type_
        else:
            return self.__class__.type_store[mangled_name]

    def ctypeInit(self, value, transfer_value):
        for i in range(len(value)):
            self.type_.ctypeInit(value[i], transfer_value[i])

    @property
    def ctype(self):
        #return ctypes.POINTER(self.type_.ctype * self.shape)
        return self.type_.ctype * self.shape

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.type_ == other.type_
                and self.shape == other.shape)

    def __ne__(self, other):
        return not self.__eq__(other)

    def loadSubscript(self, cge, container, idx):
        # TODO address calculation is same as for ArrayType, unify?
        p = cge.builder.gep(container.translate(cge),
                            [Int.constant(0), idx.translate(cge)],
                            inbounds=True)
        return p


class Callable(metaclass=ABCMeta):
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
            # bound is a reference, we don't want the * as part of the name
            assert self.bound.isReference()
            return "{}.{}".format(str(self.bound)[1:], self.name)
        else:
            return self.name

    def _llvmType(self, module):
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
        return ArrayType.fromObj(obj)
    elif type_ == list:
        return ListType.fromObj(obj)
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



# values which need to be destructed: Any complex type that is wrapped.
_stella_values_ = []

class Typable(object):
    type = NoType
    llvm = None

    def unify_type(self, tp2, debuginfo, is_reference=False):
        """Returns: widened:bool, needs_cast:bool
        widened: this type changed
        needs_cast: tp2 needs to be cast to this type
        """
        if is_reference:
            tp1 = self.type.dereference()
        else:
            tp1 = self.type
        if tp1 == tp2:
            pass
        elif tp1 == NoType:
            if is_reference:
                self.type = Reference(tp2)
            else:
                self.type = tp2
        elif tp2 == NoType:
            pass
        elif tp1 == Int and tp2 == Float:
            if is_reference:
                self.type = Reference(Float)
            else:
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

    def llvmType(self, module):
        """Map from Python types to LLVM types."""
        return self.type.llvmType(module)

    def translate(self, cge):
        # TODO assert self.llvm here? This fails with uninitialized values like
        # test.langconstr.new_global_var
        return self.llvm

    def ctype2Python(self, cge):
        pass

    def destruct(self):
        pass


class ImmutableType(object):
    def unify_type(self, tp2, debuginfo):
        raise TypeError("Type {} is immutable, it cannot be unified with {} at {}".format(
            self.type, tp2, debuginfo))

    def llvmType(self, module):
        """Map from Python types to LLVM types."""
        return self.type.llvmType(module)


class Const(Typable):
    value = None

    def __init__(self, value):
        self.value = value
        try:
            if type(value) == tuple:
                self.type = TupleType([get(v) for v in value])
            else:
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
        self.type = Reference(ArrayType.fromObj(array))
        self.value = array

    def translate(self, cge):
        ptr_int = self.value.ctypes.data
        ptr_int_llvm = Int.constant(ptr_int)

        type_ = self.type.llvmType(cge.module)
        self.llvm = ll.Constant(tp_int, ptr_int_llvm).inttoptr(type_)
        return self.llvm

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return str(self)


class Struct(Typable):
    obj_store = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        """Only one Struct representation per Python object instance.
        """
        if not hasattr(obj, '__stella_wrapper__'):
            obj.__stella_wrapper__ = Struct(obj)
            _stella_values_.append(obj.__stella_wrapper__)
        assert isinstance(obj.__stella_wrapper__, klass)
        return obj.__stella_wrapper__

    def __init__(self, obj):
        self.type = StructType.fromObj(obj)
        self.value = obj
        self.transfer_attributes = {}
        for name, type_ in self.type.items():
            if type_.req_transfer:
                attrib = getattr(obj, name)
                if id(attrib) == id(obj):
                    # self-reference TODO does it even need a transfer then?
                    # self.transfer_attributes[name] = self
                    pass
                else:
                    self.transfer_attributes[name] = wrapValue(attrib)

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return repr(self.type)

    def translate(self, cge):
        for wrapped in self.transfer_attributes.values():
            wrapped.translate(cge)
        if not self.llvm:
            (self.llvm, self.transfer_value) = self.type.constant(self.value, cge)
        assert self.transfer_value
        return self.llvm

    def ctype2Python(self, cge):
        """At the end of a Stella run, the struct's values need to be copied back
        into Python.

        Please call self.destruct() afterwards.
        """
        self.type.ctype2Python(self.transfer_value, self.value)
        for wrapped in self.transfer_attributes.values():
            wrapped.ctype2Python(cge)

    def destruct(self):
        del self.transfer_value
        del self.value.__stella_wrapper__


class List(Typable):
    _registry = {}  # Class variable

    @classmethod
    def fromObj(klass, obj):
        """Only one Struct representation per Python object instance.
        """
        if id(obj) not in klass._registry:
            wrapped = klass(obj)
            klass._registry[id(obj)] = wrapped
            _stella_values_.append(wrapped)
        return klass._registry[id(obj)]

    @classmethod
    def destructList(klass):
        klass._registry.clear()

    def __init__(self, obj):
        self.type = ListType.fromObj(obj)
        self.type.makePointer()
        self.value = obj

        self.transfer_value = self.type.ctype()

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return repr(self.type)

    def ctypeInit(self):
        self.type.ctypeInit(self.value, self.transfer_value)

    def translate(self, cge):
        if self.llvm:
            return self.llvm

        self.ctypeInit()

        addr_llvm = Int.constant(int(ctypes.addressof(self.transfer_value)))
        self.llvm = ll.Constant(tp_int, addr_llvm).inttoptr(self.type.llvmType(cge.module))
        return self.llvm

    def ctype2Python(self, cge):
        """At the end of a Stella run, all list elements need to get copied back
        into Python.

        Please call self.destruct() afterwards.
        """
        for i in range(len(self.value)):
            self.type.type_.ctype2Python(self.transfer_value[i], self.value[i])

    def destruct(self):
        del self.transfer_value

    def loadSubscript(self, cge, container, idx):
        p = cge.builder.gep(container.translate(cge),
                            [Int.constant(0), idx.translate(cge)],
                            inbounds=True)
        return p

    def storeSubscript(self, cge, container, idx, value):
        p = cge.builder.gep(
            container.translate(cge), [
                Int.constant(0), idx.translate(cge)], inbounds=True)
        cge.builder.store(value.translate(cge), p)


class Tuple(Typable):
    def __init__(self, values):
        self.values = [wrapValue(v) for v in values]
        self.type = TupleType([v.type for v in self.values])

    def __str__(self):
        return "(tuple, {} elems)".format(len(self.values))

    def __repr__(self):
        return "({})".format(", ".join( [str(v) for v in self.values]))

    def translate(self, cge):
        if self.llvm:
            return self.llvm

        # A struct is a constant and can only be initialize with constants.
        # Insert the non-constants afterwards
        init = []
        self.inserts = []
        for i, v in enumerate(self.values):
            if isinstance(v, Const):
                init.append(v.translate(cge))
            else:
                init.append(ll.Constant(v.type.llvmType(cge.module), None))
                self.inserts.append((i, v))

        self.llvm = ll.Constant.literal_struct(init)
        for i, v in self.inserts:
            self.llvm = cge.builder.insert_value(self.llvm, v.translate(cge), i)
        return self.llvm


def wrapValue(value):
    if isinstance(value, Typable):
        # already wrapped, nothing to do
        return value
    type_ = type(value)
    if supported_scalar(type_) or type_ == tuple:
        return Const(value)
    elif type_ == np.ndarray:
        return NumpyArray(value)
    elif type_ == list:
        return List.fromObj(value)
    else:
        return Struct.fromObj(value)


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
            self.llvm = cge.builder.sitofp(self.obj.llvm, Float.llvmType(cge.module), self.name)
        return self.llvm

    def __str__(self):
        return self.name


def destruct():
    global _stella_values_
    for value in _stella_values_:
        value.destruct()
    _stella_values_ = []
    FunctionType.destruct()
    StructType.destruct()
    ListType.destruct()
    CType.destruct()
    List.destructList()

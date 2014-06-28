import logging
import inspect
import weakref
import types
import sys
from abc import ABCMeta, abstractmethod
import ctypes
import math

import llvm
import llvm.core
import llvm.ee
import numpy as np

from . import exc
from . import utils
from . import tp
from .intrinsics import python


class Typable(object):
    type = tp.NoType

    def unify_type(self, tp2, debuginfo):
        tp1 = self.type
        if tp1 == tp2:
            pass
        elif tp1 == tp.NoType:
            self.type = tp2
        elif tp2 == tp.NoType:
            pass
        elif (tp1 == tp.Int and tp2 == tp.Float) or (tp1 == tp.Float and tp2 == tp.Int):
            self.type = tp.Float
            return True
        else:
            raise exc.TypingError("Unifying of types {} and {} (not yet) implemented".format(
                tp1, tp2), debuginfo)

        return False

    def llvmType(self):
        """Map from Python types to LLVM types."""
        return self.type.llvmType()


class Const(Typable):
    value = None

    def __init__(self, value):
        self.value = value
        try:
            self.type = tp.get_scalar(value)
            self.name = str(value)
        except exc.TypingError as e:
            self.name = "InvalidConst({0}, type={1})".format(value, type(value))
            raise e
        self.translate()

    def translate(self):
        self.llvm = self.type.constant(self.value)

    def unify_type(self, tp2, debuginfo):
        r = super().unify_type(tp2, debuginfo)
        if r:
            self.translate()
        return r

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class NumpyArray(Const):
    def __init__(self, array):
        assert isinstance(array, np.ndarray)

        # TODO: multi-dimensional arrays
        self.type = tp.ArrayType.fromArray(array)
        self.value = array

        self.translate()

    def translate(self):
        ptr_int = self.value.ctypes.data  # int
        ptr_int_llvm = tp.Int.constant(ptr_int)
        type_ = llvm.core.Type.pointer(self.type.llvmType())
        self.llvm = llvm.core.Constant.inttoptr(ptr_int_llvm, type_)

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return str(self)


def wrapValue(value):
    if type(value) == np.ndarray:
        return NumpyArray(value)
    else:
        return Const(value)


class Register(Typable):
    name = None

    def __init__(self, func, name=None):
        super().__init__()
        if name:
            assert type(name) == str
            self.name = name
        else:
            self.name = func.newRegisterName()

    def __str__(self):
        return "{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name


class StackLoc(Typable):
    name = None

    def __init__(self, func, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return "*{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name


class GlobalVariable(Typable):
    name = None
    initial_value = None

    def __init__(self, name, initial_value=None):
        super().__init__()
        self.name = name
        if initial_value is not None:
            self.setInitialValue(initial_value)

    def setInitialValue(self, initial_value):
        if isinstance(initial_value, Typable):
            self.initial_value = initial_value
        else:
            self.initial_value = wrapValue(initial_value)
        self.type = self.initial_value.type
        self.type.makePointer()

    def __str__(self):
        return "+{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name

    def translate(self, module, builder):
        self.llvm = module.add_global_variable(self.llvmType(), self.name)
        if hasattr(self.initial_value, 'llvm'):
            self.llvm.initializer = self.initial_value.llvm
        else:
            self.llvm.initializer = llvm.core.Constant.undef(self.initial_value.type.llvmType())


class Cast(Typable):
    def __init__(self, obj, tp):
        assert obj.type != tp
        self.obj = obj
        self.type = tp
        self.emitted = False

        logging.debug("Casting {0} to {1}".format(self.obj.name, self.type))
        self.name = "({0}){1}".format(self.type, self.obj.name)

    def translate(self, builder):
        if self.emitted:
            assert hasattr(self, 'llvm')
            return
        self.emitted = True

        # TODO: HACK: instead of failing, let's make it a noop
        # assert self.obj.type == int and self.type == float
        if self.obj.type == self.type:
            self.llvm = self.obj.llvm
            return

        if isinstance(self.obj, Const):
            value = float(self.obj.value)
            self.llvm = self.obj.type.constant(value)
        else:
            self.llvm = builder.sitofp(self.obj.llvm, tp.Float.llvmType(), self.name)

    def __str__(self):
        return self.name


@utils.linkedlist
class IR(metaclass=ABCMeta):
    args = None
    result = None
    debuginfo = None
    llvm = None
    block = None
    loc = ''
    discard = False

    def __init__(self, func, debuginfo):
        self.debuginfo = debuginfo
        self.args = []

    def addConst(self, arg):
        self.addArg(wrapValue(arg))

    def addArg(self, arg):
        self.args.append(arg)

    def addRawArg(self, arg):
        raise exc.UnimplementedError("{0}.addRawArg() is not implemented".format(
            self.__class__.__name__))

    def addLocalName(self, func, name):
        self.args.append(func.getStackLoc(name))
        # TODO: is a base implementation needed??

    def popFirstArg(self):
        first = self.args[0]
        self.args = self.args[1:]
        return first

    def cast(self, builder):
        for arg in self.args:
            if isinstance(arg, Cast):
                arg.translate(builder)

    @abstractmethod
    def stack_eval(self, func, stack):
        pass

    @abstractmethod
    def translate(self, module, builder):
        pass

    @abstractmethod
    def type_eval(self, func):
        pass

    def __str__(self):
        if self.discard:
            b = '('
            e = ')'
        else:
            b = e = ''

        return "{0}{1} {2} {3}{4}".format(
            b,
            self.__class__.__name__,
            self.result,
            ", ".join([str(v) for v in self.args]),
            e)

    def __repr__(self):
        # TODO: are there reasons not to do this?
        return self.__str__()

    def locStr(self):
        return "{0:2s} {1}".format(str(self.loc), str(self))

    def equivalent(self, other):
        """Equality but location independent.

        This method may need to be overridden by the concrete implementation.
        """
        return type(self) == type(other) and self.args == other.args


class PhiNode(IR):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

        self.blocks = []
        self.stacked = False

    def stack_eval(self, func, stack):
        tos = stack.peek()

        # sanity check: either there is always a tos or never
        if self.stacked:
            if tos and not self.result:
                raise exc.StellaException("Invalid bytecode sequence: unexpected tos")
            if not tos and self.result:
                raise exc.StellaException("Invalid bytecode sequence: expected tos")

        if tos:
            if not self.result:
                self.result = Register(func)
            self.args.append(stack.pop())
            self.blocks.append(self.args[-1].bc)
            stack.push(self.result)

        self.stacked = True

    def type_eval(self, func):
        if len(self.args) == 0:
            return
        for arg in self.args:
            self.result.unify_type(arg.type, self.debuginfo)

    def translate(self, module, builder):
        if len(self.args) == 0:
            return
        phi = builder.phi(self.result.llvmType(), self.result.name)
        for arg in self.args:
            phi.add_incoming(arg.llvm, arg.bc.block)

        self.result.llvm = phi


class Scope(object):
    """
    Used to add scope functionality to an object
    """
    def __init__(self, parent):
        self.parent = parent
        self.register_n = 0
        self.registers = dict()
        self.stacklocs = dict()

    def newRegisterName(self):
        n = str(self.register_n)
        self.register_n += 1
        return n

    def getOrNewRegister(self, name):
        if name not in self.registers:
            self.registers[name] = Register(self, name)
        return self.registers[name]

    def getRegister(self, name):
        if name not in self.registers:
            raise exc.UndefinedError(name)
        return self.registers[name]

    def getOrNewStackLoc(self, name):
        isnew = False
        if name not in self.stacklocs:
            self.stacklocs[name] = StackLoc(self, name)
            isnew = True
        return (self.stacklocs[name], isnew)

    def getStackLoc(self, name):
        if name not in self.stacklocs:
            raise exc.UndefinedError(name)
        return self.stacklocs[name]


class Globals(object):
    def __init__(self):
        self.store = dict()

    def __setitem__(self, key, value):
        # TODO: should uniqueness be enforced here?
        assert key not in self.store
        self.store[key] = value

    def __getitem__(self, key):
        if key not in self.store:
            raise exc.UndefinedGlobalError(key)
        return self.store[key]

    def all(self, tp=None):
        if tp is None:
            return self.store.items()
        else:
            return [(k, v) for k, v in self.store.items() if isinstance(v, tp)]


class Module(object):
    i = 0

    def __init__(self):
        super().__init__()
        """funcs is the set of python functions which are compiled by Stella"""
        self.funcs = set()
        self._todo = []
        self.entry = None
        self.llvm = None
        self.namestore = Globals()
        self.external_modules = dict()
        self._cleanup = []

    def addFunc(self, f):
        self.funcs.add(f)

    def makeEntry(self, f, args):
        assert self.entry is None
        self.entry = f
        self.namestore[f.name] = f
        self.entry_args = args

    def getExternalModule(self, mod):
        if mod not in self.external_modules:
            self.external_modules[mod] = ExtModule(mod)
        return self.external_modules[mod]

    def getExternalModules(self):
        return self.external_modules.values()

    def _wrapPython(self, key, item, module=None):
        if type(item) in (types.FunctionType, types.BuiltinFunctionType):
            intrinsic = getIntrinsic(item)

            if intrinsic:
                wrapped = intrinsic()
            elif hasattr(module, '__file__') and module and module.__file__[-3:] == '.so':
                ext_module = self.getExternalModule(module)
                wrapped = ext_module.getFunction(item)
            else:
                wrapped = Function(item, self)
                self.addFunc(wrapped)
        elif isinstance(item, types.ModuleType):
            # no need to wrap it, it will be used with self.loadExt()
            wrapped = item
        else:
            # Assume it is a global variable
            # TODO: is this safe? How do I catch types that aren't supported
            # without listing all valid types?
            wrapped = GlobalVariable(key, item)

        return wrapped

    def loadExt(self, module, attr):
        assert isinstance(module, types.ModuleType) and type(attr) == str

        key = module.__name__ + '.' + attr
        try:
            wrapped = self.namestore[key]
        except exc.UndefinedGlobalError as e:
            if attr not in module.__dict__:
                raise e

            item = module.__dict__[attr]
            if hasattr(module, '__file__') and module.__file__[-3:] == '.so' and \
                    type(item) == type(print):
                # external module
                pass
            elif type(item) not in (types.FunctionType, type(print)):
                raise exc.UnimplementedError(
                    "Currently only Functions can be imported (not {0})".format(type(item)))
            wrapped = self._wrapPython(key, item, module)
            self.namestore[key] = wrapped
        return wrapped

    def loadGlobal(self, func, key):
        try:
            wrapped = self.namestore[key]
        except exc.UndefinedGlobalError as e:
            # TODO: too much nesting, there should be a cleaner way to detect these types
            if key == 'len':
                item = len
            elif key not in func.f.__globals__:
                if tp.supported_scalar(key):
                    return __builtins__[key]
                else:
                    raise exc.UndefinedError("Cannot find global variable `{0}'".format(key))
                raise e
            else:
                item = func.f.__globals__[key]
            wrapped = self._wrapPython(key, item)

            self.namestore[key] = wrapped
        return wrapped

    def newGlobal(self, func, name):
        """MUST ensure that loadGlobal() fails before calling this function!"""
        wrapped = GlobalVariable(name, None)
        self.namestore[name] = wrapped
        return wrapped

    def functionCall(self, func, args, kwargs):
        if isinstance(func, Foreign):
            # no need to analyze it
            return

        if kwargs is None:
            kwargs = {}
        if not func.analyzed:
            self.todoAdd(func, args, kwargs)

    def todoAdd(self, func, args, kwargs):
        self._todo.append((func, args, kwargs))

    def todoLastFunc(self, func):
        if len(self._todo) > 0:
            (f, _, _) = self._todo[-1]
            if f == func:
                return True
        return False

    def todoNext(self):
        n = self._todo[0]
        self._todo = self._todo[1:]
        return n

    def todoCount(self):
        return len(self._todo)

    def translate(self):
        self.llvm = llvm.core.Module.new('__stella__'+str(self.__class__.i))
        self.__class__.i += 1
        for impl in self.funcs:
            impl.translate(self.llvm)

    def destruct(self):
        logging.debug("destruct() of " + repr(self))
        while True:
            try:
                d = self._cleanup.pop()
                d()
            except IndexError:
                break

    def addDestruct(self, d):
        self._cleanup.append(d)

    def __del__(self):
        logging.debug("DEL  " + repr(self))
        if len(self._cleanup) > 0:
            self.destruct()

    def getLlvmIR(self):
        if self.llvm:
            return str(self.llvm)
        else:
            return "<no code yet>"

    def __str__(self):
        return '__module__' + str(id(self))


class Callable(metaclass=ABCMeta):
    arg_defaults = {}
    arg_names = []

    @abstractmethod
    def getReturnType(self, args, kw_args):
        pass

    def getResult(self, func):
        return Register(func)

    def readSignature(self, f):
        if f == len:
            # yay, special cases
            self.arg_names = ['obj']
            self.arg_defaults = []
            return
        argspec = inspect.getargspec(f)
        self.arg_names = argspec.args
        self.arg_defaults = [Const(default) for default in argspec.defaults or []]

    def combineAndCheckArgs(self, args, kwargs):
        num_args = len(args)
        # TODO: is this the right place to check number of arguments?
        if num_args+len(kwargs) < len(self.arg_names)-len(self.arg_defaults):
            raise exc.TypingError("takes at least {0} argument(s) ({1} given)".format(
                len(self.arg_names)-len(self.arg_defaults), len(args)+len(kwargs)))
        if num_args+len(kwargs) > len(self.arg_names):
            raise exc.TypingError("takes at most {0} argument(s) ({1} given)".format(
                len(self.arg_names), len(args)))

        # just initialize r to a list of the correct length
        # TODO: I could initialize this smarter
        r = list(self.arg_names)

        # copy supplied regular arguments
        for i in range(len(args)):
            r[i] = args[i]

        # set default values
        def_offset = len(self.arg_names)-len(self.arg_defaults)
        for i in range(max(num_args, len(self.arg_names)-len(self.arg_defaults)),
                       len(self.arg_names)):
            r[i] = self.arg_defaults[i-def_offset]

        # insert kwargs
        for k, v in kwargs.items():
            try:
                idx = self.arg_names.index(k)
                if idx < num_args:
                    raise exc.TypingError("got multiple values for keyword argument '{0}'".format(
                        self.arg_names[idx]))
                r[idx] = v
            except ValueError:
                raise exc.TypingError("Function does not take an {0} argument".format(k))

        return r

    def call(self, module, builder, args, kw_args):
        args = self.combineAndCheckArgs(args, kw_args)

        return builder.call(self.llvm, [arg.llvm for arg in args])


class Function(Callable, Scope):
    def __init__(self, f, module):
        # TODO: pass the module as the parent for scope
        super().__init__(None)
        self.f = f
        self.name = f.__name__
        self.result = Register(self, '__return__')

        self.readSignature(f)
        self.args = []

        # weak reference is necessary so that Python will start garbage
        # collection for Module.
        self.module = weakref.proxy(module)

        self.analyzed = False
        self.log = logging.getLogger(str(self))

    def __str__(self):
        return self.name

    def __repr__(self):
        return super().__repr__()[:-1]+':'+str(self)+'>'

    def nameAndType(self):
        return self.name + "(" + str(self.args) + ")"

    def getReturnType(self, args, kw_args):
        return self.result.type

    def analyzeAgain(self):
        """Pushes the current function on the module's analysis todo list"""
        if not self.module.todoLastFunc(self):
            self.module.todoAdd(self, None, None)

    def loadGlobal(self, key):
        return self.module.loadGlobal(self, key)

    def newGlobal(self, key):
        return self.module.newGlobal(self, key)

    def makeEntry(self, args, kwargs):
        self.module.makeEntry(self, self.combineAndCheckArgs(args, kwargs))

    def setParamTypes(self, args, kwargs):
        combined = self.combineAndCheckArgs(args, kwargs)

        self.arg_transfer = []

        for i in range(len(combined)):
            # TODO: I don't particularly like this isinstance check here but it seems the easiest
            #       way to also handle the initial entry function
            if isinstance(combined[i], Typable):
                type_ = combined[i].type
            else:
                type_ = tp.get(combined[i])

            if isinstance(type_, tp.ArrayType):
                # TODO: create superclass for complex types
                arg = self.getOrNewRegister(self.arg_names[i])
                arg.type = type_
                arg.type.makePointer()
            else:
                name = self.arg_names[i]
                arg = self.getOrNewRegister('__param_'+name)
                arg.type = type_
                self.arg_transfer.append(name)

            self.args.append(arg)

        self.analyzed = True

    def translate(self, module):
        self.arg_types = [arg.llvmType() for arg in self.args]

        func_tp = llvm.core.Type.function(self.result.type.llvmType(), self.arg_types)
        self.llvm = module.add_function(func_tp, self.name)

        for i in range(len(self.args)):
            self.llvm.args[i].name = self.args[i].name
            self.args[i].llvm = self.llvm.args[i]

    def remove(self, bc):

        # TODO: should any of these .next become .linearNext()?
        if bc == self.bytecodes:
            self.bytecodes = bc.next

        if bc in self.incoming_jumps:
            bc_next = bc.next
            if not bc_next and bc._block_parent:
                bc_next = bc._block_parent.next
                # _block_parent will be move with bc.remove() below
            assert bc_next
            self.incoming_jumps[bc_next] = self.incoming_jumps[bc]
            for bc_ in self.incoming_jumps[bc_next]:
                bc_.updateTargetBytecode(bc, bc_next)
            del self.incoming_jumps[bc]
        bc.remove()


class Foreign(object):
    """Mixin: This is not a Python function. It does not need to get analyzed."""
    pass
# Intrinsics {...


class Intrinsic(Foreign, Callable):
    py_func = None

    @abstractmethod
    def call(self, module, builder, args, kw_args):
        """args and kw_args are already added by a call through addArgs()"""
        pass


class Zeros(Intrinsic):
    py_func = python.zeros

    def __init__(self):
        self.readSignature(self.py_func)

    def getReturnType(self, args, kw_args):
        combined = self.combineAndCheckArgs(args, kw_args)
        shape = combined[0].value
        type_ = tp.get_scalar(combined[1])
        if not tp.supported_scalar(type_):
            raise exc.TypingError("Invalid array element type {0}".format(type_))
        return tp.ArrayType(type_, shape)

    def call(self, module, builder, args, kw_args):
        type_ = self.getReturnType(args, kw_args).llvmType()
        return builder.alloca(type_)


class Len(Intrinsic):
    """
    Determine the length of the array based on its type.
    """
    py_func = len

    def __init__(self):
        self.readSignature(self.py_func)

    def getReturnType(self, args, kw_args):
        return tp.Int

    def getResult(self, func):
        # we need the reference to back-patch
        self.result = Const(0)
        return self.result

    def call(self, module, builder, args, kw_args):
        obj = args[0]
        if not isinstance(obj.type, tp.ArrayType):
            raise exc.TypingError("Invalid array type {0}".format(self.obj.type))
        self.result.value = obj.type.shape
        self.result.translate()
        return self.result.llvm


class Log(Intrinsic):
    py_func = math.log
    intr = llvm.core.INTR_LOG

    def __init__(self):
        self.readSignature(self.py_func)

    def readSignature(self, f):
        # arg, inspect.getargspec(f) doesn't work for C/cython functions
        self.arg_names = ['x']
        self.arg_defaults = []

    def getReturnType(self, args, kw_args):
        return tp.Float

    def call(self, module, builder, args, kw_args):
        if args[0].type == tp.Int:
            args[0] = Cast(args[0], tp.Float)
            args[0].translate(builder)

        llvm_f = llvm.core.Function.intrinsic(module, self.intr, [args[0].llvmType()])
        result = builder.call(llvm_f, [args[0].llvm])
        return result

    def getResult(self, func):
        return Register(func)


class Exp(Log):
    py_func = math.exp
    intr = llvm.core.INTR_EXP

    def __init__(self):
        super().__init__()

# --

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


def getIntrinsic(func):
    if func in func2klass:
        return func2klass[func]
    else:
        return None

# } Intrinsics


class ExtModule(object):
    python = None
    signatures = {}
    funcs = dict()

    def __init__(self, python):
        assert type(python) == type(sys)

        self.python = python
        self.signatures = python.getCSignatures()

        for name, sig in self.signatures.items():
            self.funcs[name] = ExtFunction(name, sig)
        self.translated = False

    def getFile(self):
        return self.python.__file__

    def getSymbols(self):
        return self.signatures.keys()

    def getSignatures(self):
        return self.signatures.items()

    def getFunction(self, f):
        return self.funcs[f.__name__]

    def __str__(self):
        return str(self.python)

    def translate(self, module):
        if not self.translated:
            self.translated = True
            logging.debug("Adding external module {0}".format(self.python))
            clib = ctypes.cdll.LoadLibrary(self.python.__file__)
            for func in self.funcs.values():
                func.translate(clib, module)


class ExtFunction(Foreign, Callable):
    llvm = None
    name = '?()'

    def __init__(self, name, signature):
        self.name = name
        ret, arg_types = signature
        self.return_type = tp.from_ctype(ret)
        self.arg_types = list(map(tp.from_ctype, arg_types))
        self.readSignature(None)

    def __str__(self):
        return self.name

    def readSignature(self, f):
        # arg, inspect.getargspec(f) doesn't work for C/cython functions
        self.arg_names = ['arg{0}' for i in range(len(self.arg_types))]
        self.arg_defaults = []

    def getReturnType(self, args, kw_args):
        return self.return_type

    def translate(self, clib, module):
        logging.debug("Adding external function {0}".format(self.name))
        f = getattr(clib, self.name)
        llvm.ee.dylib_add_symbol(self.name, ctypes.cast(f, ctypes.c_void_p).value)

        llvm_arg_types = [arg.llvmType() for arg in self.arg_types]

        func_tp = llvm.core.Type.function(self.return_type.llvmType(), llvm_arg_types)
        self.llvm = module.add_function(func_tp, self.name)

    def call(self, module, builder, args, kw_args):
        args = self.combineAndCheckArgs(args, kw_args)

        args_llvm = []
        for arg, arg_type in zip(args, self.arg_types):
            if arg.type != arg_type:
                # TODO: trunc is not valid for all type combinations.
                # Needs to be generalized.
                llvm = builder.trunc(arg.llvm, arg_type.llvmType(), '({0}){1}'.format(
                    arg_type, arg.name))
            else:
                llvm = arg.llvm
            args_llvm.append(llvm)

        return builder.call(self.llvm, args_llvm)

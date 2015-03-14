import logging
import weakref
import types
import sys
from abc import ABCMeta, abstractmethod
import ctypes
import inspect

import llvmlite.ir as ll
import llvmlite.binding as llvm

from . import exc
from . import utils
from . import tp
from .storage import Register, StackLoc, GlobalVariable
from . import intrinsics


@utils.linkedlist
class IR(metaclass=ABCMeta):
    args = None
    stack_args = None
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
        self.addArg(tp.wrapValue(arg))

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

    def cast(self, cge):
        for arg in self.args:
            if isinstance(arg, tp.Cast):
                arg.translate(cge)

    def grab_stack(self):
        """
        Call first during type evaluation. Gets the results from the stack and
        adds them to args.
        """
        if self.stack_args:
            for arg in self.stack_args:
                # TODO should arg.result always be a list?
                if isinstance(arg.result, list):
                    result = arg.result.pop()
                    # keep the result, we need it for retyping
                    arg.result.insert(0, result)
                else:
                    result = arg.result
                result.bc = arg
                self.args.append(result)
            self.stack_args = None

    @abstractmethod
    def stack_eval(self, func, stack):
        pass

    @abstractmethod
    def translate(self, cge):
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
        return "{0:3s} {1}".format(str(self.loc), str(self))

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
        self.stack_args = []

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
            self.stack_args.append(stack.pop())
            self.blocks.append(self.stack_args[-1])
            stack.push(self)

        self.stacked = True

    def type_eval(self, func):
        self.grab_stack()
        if len(self.args) == 0:
            return
        for arg in self.args:
            self.result.unify_type(arg.type, self.debuginfo)

    def translate(self, cge):
        if len(self.args) == 0:
            return
        phi = cge.builder.phi(self.result.llvmType(cge.module), self.result.name)
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
        self._todo = []
        self.entry = None
        self.llvm = None
        self.namestore = Globals()
        self.external_modules = dict()
        self._cleanup = []

    def _getFunction(self, item):
        if isinstance(item, tp.FunctionType):
            f_type = item
        else:
            f_type = tp.get(item)
        try:
            f = self.namestore[f_type.name]
        except exc.UndefinedGlobalError:
            f = Function(f_type, self)
            self.namestore[f_type.name] = f

        return f

    def getFunctionRef(self, item):
        if isinstance(item, Function):
            f = item
        else:
            f = self._getFunction(item)
        if f.type_.bound:
            return BoundFunctionRef(f)
        else:
            return FunctionRef(f)

    def makeEntry(self, funcref, args):
        assert self.entry is None
        self.entry = funcref
        self.entry_args = args

    def getExternalModule(self, mod):
        if mod not in self.external_modules:
            self.external_modules[mod] = ExtModule(mod)
        return self.external_modules[mod]

    def getExternalModules(self):
        return self.external_modules.values()

    def _wrapPython(self, key, item, module=None):
        if isinstance(item, (types.FunctionType, types.BuiltinFunctionType)):
            intrinsic = getIntrinsic(item)

            if intrinsic:
                wrapped = intrinsic
            elif module and hasattr(module, '__file__') and module.__file__[-3:] == '.so':
                ext_module = self.getExternalModule(module)
                wrapped = ext_module.getFunctionRef(item)
            else:
                f = self._getFunction(item)
                wrapped = self.getFunctionRef(f)
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
            elif not isinstance(item, (types.FunctionType, type(print))):
                raise exc.UnimplementedError(
                    "Currently only Functions can be imported (not {0})".format(type(item)))
            wrapped = self._wrapPython(key, item, module)
            self.namestore[key] = wrapped

        # instrinsinc check: we need a new instance for every call!
        # TODO: see also self.loadGlobal()
        if inspect.isclass(wrapped) and issubclass(wrapped, intrinsics.Intrinsic):
            return wrapped()
        else:
            return wrapped

    def loadGlobal(self, func, key):
        try:
            wrapped = self.namestore[key]
            if isinstance(wrapped, Function):
                wrapped = self.getFunctionRef(wrapped)
        except exc.UndefinedGlobalError as e:
            # TODO: too much nesting, there should be a cleaner way to detect these types
            if key == 'len':
                item = len
            elif key not in func.pyFunc().__globals__:
                if tp.supported_scalar_name(key):
                    return __builtins__[key]
                else:
                    raise exc.UndefinedError("Cannot find global variable `{0}'".format(key))
                raise e
            else:
                item = func.pyFunc().__globals__[key]
            wrapped = self._wrapPython(key, item)

            # _wrapPython will create an entry for functions _only_
            if not isinstance(wrapped, FunctionRef):
                self.namestore[key] = wrapped

        # instrinsinc check: we need a new instance for every call!
        # TODO: this may be required in _getFunction?
        if inspect.isclass(wrapped) and issubclass(wrapped, intrinsics.Intrinsic):
            return wrapped()
        else:
            return wrapped

    def newGlobal(self, func, name):
        """MUST ensure that loadGlobal() fails before calling this function!"""
        wrapped = GlobalVariable(name, None)
        self.namestore[name] = wrapped
        return wrapped

    def functionCall(self, funcref, args, kwargs):
        func = funcref.function
        if isinstance(func, tp.Foreign):
            # no need to analyze it
            return

        if kwargs is None:
            kwargs = {}
        if not func.analyzed:
            self.todoAdd(funcref, args, kwargs)

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
        self.llvm = ll.Module('__stella__'+str(self.__class__.i), context=ll.context.Context())
        self.__class__.i += 1
        for _, impl in self.namestore.all(Function):
            impl.translate(self)

    def destruct(self):
        """Clean up this objects so that gc will succeed.

        Function has a weakref back to Module, but something in Function
        confuses the gc algorithm and Module will never be collected while
        fully intact.
        """
        logging.debug("destruct() of " + repr(self))

        # destruct() can be called more than once
        if hasattr(self, 'entry'):
            del self.entry
        if hasattr(self, 'entry_args'):
            del self.entry_args

        msg = []
        while True:
            try:
                d = self._cleanup.pop()
                msg.append(str(d))
                d()
            except IndexError:
                break
        if len(msg) > 0:
            logging.debug("Called destructors: " + ", ".join(msg))

    def addDestruct(self, d):
        self._cleanup.append(d)

    def __del__(self):
        logging.debug("DEL  " + repr(self))
        self.destruct()

    def getLlvmIR(self):
        if self.llvm:
            return str(self.llvm)
        else:
            return "<no code yet>"

    def __str__(self):
        return '__module__' + str(id(self))


class Function(Scope):
    """
    Represents the code of the function. Has to be unique for each source
    instance.
    """
    def __init__(self, type_, module):
        # TODO: pass the module as the parent for scope
        super().__init__(None)
        if not isinstance(type_, tp.FunctionType):
            type_ = tp.FunctionType.get(type_)
        self.type_ = type_
        self.name = type_.fq
        self.result = Register(self, '__return__')

        self.args = []

        # Use a weak reference here because module has a reference to the
        # function -- the cycle would prevent gc
        self.module = weakref.proxy(module)

        self.analyzed = False
        self.log = logging.getLogger(str(self))

    def pyFunc(self):
        return self.type_.pyFunc()

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}:{}>".format(super().__repr__()[:-1], self)

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

    def setupArgs(self, args, kwargs):
        tp_args = [arg.type for arg in args]
        tp_kwargs = {k: v.type for k, v in kwargs.items()}

        combined = self.type_.typeArgs(tp_args, tp_kwargs)

        self.arg_transfer = []

        for i in range(len(combined)):
            type_ = combined[i]

            if type_.on_heap:
                # TODO: create superclass for complex types
                arg = self.getOrNewRegister(self.type_.arg_names[i])
                arg.type = type_
            else:
                name = self.type_.arg_names[i]
                arg = self.getOrNewRegister('__param_'+name)
                arg.type = type_
                self.arg_transfer.append(name)

            self.args.append(arg)

        self.analyzed = True

    def translate(self, module):
        self.arg_types = [arg.llvmType(module) for arg in self.args]

        func_tp = ll.FunctionType(self.result.type.llvmType(module), self.arg_types)
        self.llvm = ll.Function(module.llvm, func_tp, name=self.name)

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


class FunctionRef(tp.Callable):
    def __init__(self, function):
        self.function = function

    def __str__(self):
        return "<*function {}>".format(self.function)

    def __repr__(self):
        return "{}:{}>".format(super().__repr__()[:-1], self.function.type_.fq)

    @property
    def type_(self):
        """Convenience function to return the referenced function's type."""
        # TODO: this may be confusing if function references become first class
        # citizens!
        return self.function.type_

    def makeEntry(self, args, kwargs):
        # TODO Verify that type checking occurred at this point
        self.function.module.makeEntry(self, self.combineArgs(args, kwargs))

    def getResult(self, func):
        return Register(func)

    # TODO: Maybe the caller of the following functions should resolve to
    # self.function instead of making a proxy call here.

    def getReturnType(self, args, kw_args):
        return self.function.getReturnType(args, kw_args)

    @property
    def result(self):
        return self.function.result

    @property
    def llvm(self):
        return self.function.llvm


class BoundFunctionRef(FunctionRef):
    def __init__(self, function):
        super().__init__(function)

    def __str__(self):
        return "<*bound method {} of {}>".format(self.function, self.type_.bound)

    @property
    def self_type(self):
        return self.type_.bound

    @property
    def f_self(self):
        return self._f_self

    @f_self.setter
    def f_self(self, value):
        # TODO: These should be throw-away objects. I want to know if they live
        # longer than expected.
        assert not hasattr(self, '_f_self')
        self._f_self = value

    def combineArgs(self, args, kwargs):
        full_args = [self.f_self] + args
        return super().combineArgs(full_args, kwargs)


def getIntrinsic(func):
    if func in intrinsics.func2klass:
        return intrinsics.func2klass[func]
    else:
        return None


class ExtModule(object):
    python = None
    signatures = {}
    funcs = dict()

    def __init__(self, python):
        assert type(python) == type(sys)

        self.python = python
        self.signatures = python.getCSignatures()

        for name, sig in self.signatures.items():
            type_ = tp.ExtFunctionType(sig)
            self.funcs[name] = ExtFunction(name, type_)
        self.translated = False

    def getFile(self):
        return self.python.__file__

    def getSymbols(self):
        return self.signatures.keys()

    def getSignatures(self):
        return self.signatures.items()

    def getFunction(self, f):
        return self.funcs[f.__name__]

    def getFunctionRef(self, f):
        return ExtFunctionRef(self.funcs[f.__name__])

    def __str__(self):
        return str(self.python)

    def translate(self, module):
        if not self.translated:
            self.translated = True
            logging.debug("Adding external module {0}".format(self.python))
            clib = ctypes.cdll.LoadLibrary(self.python.__file__)
            for func in self.funcs.values():
                func.translate(clib, module)


class ExtFunction(object):
    llvm = None
    analyzed = True
    name = '?()'

    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_
        self.result = Register(self, '__return__')

    def __str__(self):
        return self.name

    def getReturnType(self, args, kw_args):
        # TODO: Do we need to type self.result?
        return self.type_.getReturnType(args, kw_args)

    def translate(self, clib, module):
        logging.debug("Adding external function {0}".format(self.name))
        f = getattr(clib, self.name)
        llvm.add_symbol(self.name, ctypes.cast(f, ctypes.c_void_p).value)

        llvm_arg_types = [arg.llvmType(module) for arg in self.type_.arg_types]

        func_tp = ll.FunctionType(self.type_.return_type.llvmType(module), llvm_arg_types)
        self.llvm = ll.Function(module, func_tp, self.name)


class ExtFunctionRef(tp.Callable):
    def __init__(self, function):
        self.function = function

    @property
    def type_(self):
        return self.function.type_

    def getReturnType(self, args, kwargs):
        return self.function.getReturnType(args, kwargs)

    def getResult(self, func):
        return Register(func)

    def call(self, cge, args, kw_args):
        args = self.combineArgs(args, kw_args)

        args_llvm = []
        for arg, arg_type in zip(args, self.type_.arg_types):
            if arg.type != arg_type:
                # TODO: trunc is not valid for all type combinations.
                # Needs to be generalized.
                llvm = cge.builder.trunc(arg.translate(cge),
                                         arg_type.llvmType(cge.module),
                                         '({0}){1}'.format(arg_type, arg.name))
            else:
                llvm = arg.llvm
            args_llvm.append(llvm)

        return cge.builder.call(self.function.llvm, args_llvm)

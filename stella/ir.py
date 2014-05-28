import logging
import inspect
import weakref
import types
from abc import ABCMeta, abstractmethod, abstractproperty

from .llvm import *
from .exc import *
from .utils import *
from .intrinsics import *

import pdb

class Typable(object):
    type = NoType
    def unify_type(self, tp2, debuginfo):
        tp1 = self.type
        if   tp1 == tp2:    pass
        elif tp1 == NoType: self.type = tp2
        elif tp2 == NoType: pass
        elif (tp1 == int and tp2 == float) or (tp1 == float and tp2 == int):
            self.type = float
            return True
        else:
            raise TypingError ("Unifying of types " + str(tp1) + " and " + str(tp2) + " not yet implemented", debuginfo)

        return False

class Register(Typable):
    name = None

    def __init__(self, func, name = None):
        super().__init__()
        if name:
            assert type(name) == str
            self.name = name
        else:
            self.name = func.newRegisterName()

    def __str__(self):
        return "{0}<{1}>".format(self.name, self.type.__name__)
    def __repr__(self):
        return self.name

class StackLoc(Typable):
    name = None

    def __init__(self, func, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return "*{0}<{1}>".format(self.name, self.type.__name__)
    def __repr__(self):
        return self.name

class GlobalVariable(Typable):
    name = None
    initial_value = None

    def __init__(self, initial_value, name):
        super().__init__()
        self.name = name
        self.initial_value = initial_value
        self.type = type(initial_value)

    def __str__(self):
        return "+{0}<{1}>".format(self.name, self.type.__name__)
    def __repr__(self):
        return self.name


    def translate(self, module, builder):
        self.llvm = module.add_global_variable(py_type_to_llvm(self.type), self.name)
        self.llvm.initializer = llvm_constant(self.initial_value) #Constant.undef(tp)

class Const(Typable):
    value = None

    def __init__(self, value):
        self.value = value
        self.type = type(value)
        self.name = str(value)
        self.translate()

    def translate(self):
        self.llvm = llvm_constant(self.value)

    def unify_type(self, tp2, debuginfo):
        r = super().unify_type(tp2, debuginfo)
        if r:
            self.translate()
        return r

    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return self.__str__()

class Cast(object):
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

        #import pdb; pdb.set_trace()

        # TODO: HACK: instead of failing, let's make it a noop
        #assert self.obj.type == int and self.type == float
        if self.obj.type ==  self.type:
            self.llvm = self.obj.llvm
            return

        # if value attribute is present, then it is a Const
        if hasattr(self.obj, 'value'):
            self.value = float(self.obj.value)
            self.llvm = llvm_constant(self.value)
        else:
            self.llvm = builder.sitofp(self.obj.llvm, py_type_to_llvm(float), self.name)

    def __str__(self):
        return self.name

@linkedlist
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
        #import pdb; pdb.set_trace()
        self.addArg(Const(arg))

    def addArg(self, arg):
        self.args.append(arg)

    def addRawArg(self, arg):
        raise UnimplementedError("{0}.addRawArg() is not implemented".format(self.__class__.__name__))

    def addLocalName(self, func, name):
        #self.args.append(func.getRegister(name))
        self.args.append(func.getStackLoc(name))
        # TODO: is a base implementation needed??

    def popFirstArg(self):
        first = self.args[0]
        self.args = self.args[1:]
        return first

    def cast(self, builder):
        #import pdb; pdb.set_trace()
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
                raise StellaException("Invalid bytecode sequence: unexpected tos")
            if not tos and self.result:
                raise StellaException("Invalid bytecode sequence: expected tos")

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
        phi = builder.phi(py_type_to_llvm(self.result.type), self.result.name)
        for arg in self.args:
            phi.add_incoming(arg.llvm, arg.bc.block)

        self.result.llvm = phi
        #import pdb; pdb.set_trace()

class Scope(object):
    """
    Used to add scope functionality to an object
    """
    def __init__(self, parent):
        #import pdb; pdb.set_trace()
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
            raise UndefinedError(name)
        return self.registers[name]

    def getOrNewStackLoc(self, name):
        isnew = False
        if name not in self.stacklocs:
            self.stacklocs[name] = StackLoc(self, name)
            isnew = True
        return (self.stacklocs[name], isnew)

    def getStackLoc(self, name):
        if name not in self.stacklocs:
            raise UndefinedError(name)
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
            raise UndefinedGlobalError(key)
        return self.store[key]

    def all(self, tp=None):
        if tp == None:
            return self.store.items()
        else:
            return [(k,v) for k,v in self.store.items() if isinstance(v, tp)]

class Module(object):
    i=0
    def __init__(self):
        super().__init__()
        """funcs is the set of python functions which are compiled by Stella"""
        self.funcs = set()
        self._todo = []
        self.entry = None
        self.llvm = None
        self.namestore = Globals()

    def addFunc(self, f):
        self.funcs.add(f)

    def makeEntry(self, f, args):
        assert self.entry == None
        self.entry = f
        self.namestore[f.name] = f
        self.entry_args = args

    def _wrapPython(self, key, item):
        if type(item) == types.FunctionType:
            wrapped = Function(item, self)
            self.addFunc(wrapped)
        elif type(item) == types.ModuleType:
            # no need to wrap it, it will be used with self.loadExt()
            wrapped = item
        else:
            # Assume it is a global variable
            # TODO: is this safe? How do I catch types that aren't supported
            # without listing all valid types?
            wrapped = GlobalVariable(item, key)

        return wrapped

    def loadExt(self, module, attr):
        assert type(module) == types.ModuleType and type(attr) == str

        key = module.__name__ +'.'+ attr
        try:
            wrapped = self.namestore[key]
        except UndefinedGlobalError as e:
            try:
                item = module.__dict__[attr]
                if type(item) != types.FunctionType:
                    raise UnimplementedError("Currently only Functions can be imported (not {0})".format(type(item)))
                wrapped = self._wrapPython(key, item)
                self.namestore[key] = wrapped
            except KeyError:
                raise e
        return wrapped

    def loadGlobal(self, func, key):
        try:
            wrapped = self.namestore[key]
        except UndefinedGlobalError as e:
            if key not in func.f.__globals__:
                # TODO: too much nesting, there should be a cleaner way to detect these types
                if supported_py_type(key):
                    return __builtins__[key]
                else:
                    raise UnimplementedError("Type {0} is not supported".format(key))
                raise e
            item = func.f.__globals__[key]
            intrinsic = getIntrinsic(item)

            if intrinsic:
                wrapped = intrinsic
            else:
                wrapped = self._wrapPython(key, item)

            self.namestore[key] = wrapped
        return wrapped

    def functionCall(self, func, args, kwargs):
        #pdb.set_trace()
        if kwargs == None:
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
#        logging.debug("TEST {0} {1}, {2}, {3}".format(repr(self), '__stella__'+str(self.__class__.i),
#            repr(self.llvm), repr(self.llvm._ptr)))
        self.__class__.i += 1
        for impl in self.funcs:
            impl.translate(self.llvm)

#    def __del__(self):
#        logging.debug("DEL  " + repr(self))
    def getLlvmIR(self):
        if self.llvm:
            return str(self.llvm)
        else:
            return "<no code yet>"

    def __str__(self):
        return '__module__' + str(id(self))

class Function(Scope):
    def __init__(self, f, module):
        # TODO: pass the module as the parent for scope
        super().__init__(None)
        self.f = f
        self.name = f.__name__
        self.result = Register(self, '__return__')

        argspec = inspect.getargspec(f)
        self.arg_names = [n for n in argspec.args]
        self.args = [self.getOrNewRegister('__param_'+n) for n in argspec.args]
        self.arg_names = argspec.args
        self.arg_defaults = [Const(default) for default in argspec.defaults or []]
        #self.arg_values = None

        # weak reference is necessary so that Python will start garbage
        # collection for Module.
        self.module = weakref.proxy(module)

        self.analyzed = False
        self.log = logging.getLogger(str(self))

    def __str__(self):
        return self.name

    def nameAndType(self):
        return self.name + "(" + str(self.args) + ")"

    def getReturnType(self):
        return self.result.type

    def analyzeAgain(self):
        if not self.module.todoLastFunc(self):
            self.module.todoAdd(self, None, None)

    def loadGlobal(self, key):
        return self.module.loadGlobal(self, key)

    def combineAndCheckArgs(self, args, kwargs):
        def_start = len(args)
        # TODO: is this the right place to check number of arguments?
        if def_start+len(kwargs) < len(self.args)-len(self.arg_defaults):
            raise TypingError("takes at least {0} argument(s) ({1} given)".format(
                len(self.args)-len(self.arg_defaults), len(args)+len(kwargs)))
        if def_start+len(kwargs) > len(self.args):
            raise TypingError("takes at most {0} argument(s) ({1} given)".format(
                len(self.args), len(args)))

        # just initialize r to a list of the correct length
        # TODO: I could initialize this smarter
        r = list(self.arg_names)

        # copy supplied regular arguments
        for i in range(len(args)):
            r[i] = args[i]

        # set default values
        for i in range(def_start,len(self.args)):
            #logging.debug("default argument {0} has type {1}".format(i,type(self.arg_defaults[i-def_start])))
            r[i] = self.arg_defaults[i-def_start]

        # insert kwargs
        for k,v in kwargs.items():
            try:
                idx = self.arg_names.index(k)
                if idx < def_start:
                    raise TypingError("got multiple values for keyword argument '{0}'".format(self.arg_names[idx]))
                r[idx] = v
            except ValueError:
                raise TypingError("Function does not take an {0} argument".format(k))

        return r


    def makeEntry(self, args, kwargs):
        self.module.makeEntry(self, self.combineAndCheckArgs(args, kwargs))
        #self.arg_values = 

    def setParamTypes(self, args, kwargs):
        combined = self.combineAndCheckArgs(args, kwargs)

        for i in range(len(combined)):
            # TODO: I don't particularly like this isinstance check here but it seems the easiest
            #       way to also handle the initial entry function
            if isinstance(combined[i], Typable):
                self.args[i].type = combined[i].type
            else:
                self.args[i].type = type(combined[i])
        self.analyzed = True

    def translate(self, module):
        self.arg_types = [py_type_to_llvm(arg.type) for arg in self.args]

        func_tp = llvm.core.Type.function(py_type_to_llvm(self.result.type), self.arg_types)
        self.llvm = module.add_function(func_tp, self.name)

        for i in range(len(self.args)):
            self.llvm.args[i].name = self.args[i].name
            self.args[i].llvm = self.llvm.args[i]

    def remove(self, bc):
        #import pdb; pdb.set_trace()

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


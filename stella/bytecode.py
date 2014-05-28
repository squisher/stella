import dis
import logging
import sys
import inspect
import weakref
import types

import pdb

from .llvm import *
from .exc import *
from .utils import *
from .intrinsics import *
from abc import ABCMeta, abstractmethod, abstractproperty

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


def pop_stack(n):
    """
    Decorator, it takes n items off the stack
    and adds them as bytecode arguments.
    """
    def extract_n(f):
        def extract_from_stack(self, func, stack):
            args = []
            for i in range(n):
                args.append(stack.pop())
            args.reverse()

            if self.args == None:
                self.args = args
            else:
                self.args.extend(args)
            return f(self, func, stack)
        return extract_from_stack
    return extract_n

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

class BlockTerminal(object):
    """
    Marker class for instructions which terminate a block.
    """
    pass

class Poison(object):
    """
    Require that this bytecode is rewritten by bailing out
    if it is ever evaluated.

    Note that if the child overrides all methods, this mixin will be useless
    and should be removed from the child.
    """
    def stack_eval(self, func, stack):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))

    def translate(self, module, builder):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))

    def type_eval(self, func):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))


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


class Bytecode(IR):
    """
    Parent class for all Python bytecodes
    """
    pass

class LOAD_FAST(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        assert type(self.args[0]) == StackLoc
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        self.result.type = self.args[0].type

    def translate(self, module, builder):
        #tp = py_type_to_llvm(self.loc.type)
        self.result.llvm = builder.load(self.args[0].llvm)

class STORE_FAST(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.new_allocate = False

    def addLocalName(self, func, name):
        # Python does not allocate new names, it just refers to them
        #import pdb; pdb.set_trace()
        (var, self.new_allocate) = func.getOrNewStackLoc(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        self.result = self.popFirstArg()

    def type_eval(self, func):
        #func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        arg = self.args[0]
        #import pdb; pdb.set_trace()
        tp_changed = self.result.unify_type(arg.type, self.debuginfo)
        if tp_changed:
            # TODO: can I avoid a retype in some cases?
            func.retype()
            if self.result.type != arg.type:
                self.args[0] = Cast(arg, self.result.type)

    def translate(self, module, builder):
        self.cast(builder)
        if self.new_allocate:
            tp = py_type_to_llvm(self.args[0].type)
            self.result.llvm = builder.alloca(tp, name=self.result.name)
        builder.store(self.args[0].llvm, self.result.llvm)

class STORE_GLOBAL(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        # Python does not allocate new names, it just refers to them
        #import pdb; pdb.set_trace()
        var = func.loadGlobal(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        self.result = self.popFirstArg()

    def type_eval(self, func):
        #func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        arg = self.args[0]
        #import pdb; pdb.set_trace()
        tp_changed = self.result.unify_type(arg.type, self.debuginfo)
        if tp_changed:
            # TODO: can I avoid a retype in some cases?
            func.retype()
            if self.result.type != arg.type:
                self.args[0] = Cast(arg, self.result.type)

    def translate(self, module, builder):
        # Assume that the global has been allocated already.
        self.cast(builder)
        builder.store(self.args[0].llvm, self.result.llvm)

class LOAD_CONST(Bytecode):
    discard = True
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        if self.result == None:
            self.result = self.popFirstArg()
        stack.push(self.result)

    def type_eval(self, func):
        pass
    def translate(self, module, builder):
        pass

class BinaryOp(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self.result)

    def type_eval(self, func):
        for i in range(len(self.args)):
            arg = self.args[i]
            tp_changed = self.result.unify_type(arg.type, self.debuginfo)
            if tp_changed:
                if self.result.type != arg.type:
                    self.args[i] = Cast(arg, self.result.type)
                # TODO: can I avoid a retype in some cases?
                func.retype()

    def builderFuncName(self):
        try:
            return self.b_func[self.result.type]
        except KeyError:
            raise TypingError("{0} does not yet implement type {1}".format(self.__class__.__name__, self.result.type))

    def translate(self, module, builder):
        self.cast(builder)
        f = getattr(builder, self.builderFuncName())
        self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

    @abstractproperty
    def b_func(self):
        return {}

class BINARY_ADD(BinaryOp):
    b_func = {float: 'fadd', int: 'add'}

class BINARY_SUBTRACT(BinaryOp):
    b_func = {float: 'fsub', int: 'sub'}

class BINARY_MULTIPLY(BinaryOp):
    b_func = {float: 'fmul', int: 'mul'}

class BINARY_MODULO(BinaryOp):
    b_func = {float: 'frem', int: 'srem'}

class BINARY_POWER(BinaryOp):
    b_func = {float: llvm.core.INTR_POW, int: llvm.core.INTR_POWI}

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        # TODO if args[1] is int but negative, then the result will be float, too!
#        if self.args[0].type == int and self.args[1].type == int:
#            tp = int
#        else:
#            tp = float
#        tp_changed = self.result.unify_type(tp, self.debuginfo)
        super().type_eval(func)

    def translate(self, module, builder):
        # llvm.pow[i]'s first argument always has to be float
        if self.args[0].type == int:
            self.args[0] = Cast(self.args[0], float)

        self.cast(builder)

        if self.args[1].type == int:
            # powi takes a i32 argument
            power = builder.trunc(self.args[1].llvm, tp_int32, '(i32)'+self.args[1].name)
        else:
            power = self.args[1].llvm

        llvm_pow = llvm.core.Function.intrinsic(module, self.b_func[self.args[1].type], [py_type_to_llvm(self.args[0].type)])
        pow_result = builder.call(llvm_pow, [self.args[0].llvm, power])

        if isinstance(self.args[0], Cast) and self.args[0].obj.type == int and self.args[1].type == int:
            # cast back to an integer
            self.result.llvm = builder.fptosi(pow_result, py_type_to_llvm(int))
        else:
            self.result.llvm = pow_result

class BINARY_FLOOR_DIVIDE(BinaryOp):
    """Python compliant `//' operator: Slow since it has to perform type conversions and floating point division for integers"""
    b_func = {float: 'fdiv', int: 'fdiv'} # NOT USED, but required to make it a concrete class

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def type_eval(self, func):
        for arg in self.args:
            # TODO: this is a HACK
            if isinstance(arg, Cast):
                tp = arg.obj.type
            else:
                tp = arg.type
            self.result.unify_type(tp, self.debuginfo)

        # convert all arguments to float, since fp division is required to apply floor
        for i in range(len(self.args)):
            arg = self.args[i]
            do_cast = arg.type != float
            if do_cast:
                self.args[i] = Cast(arg, float)
                func.retype()

    def translate(self, module, builder):
        self.cast(builder)

        tmp = builder.fdiv(self.args[0].llvm, self.args[1].llvm, self.result.name)
        llvm_floor = llvm.core.Function.intrinsic(module, llvm.core.INTR_FLOOR, [py_type_to_llvm(float)])
        self.result.llvm = builder.call(llvm_floor, [tmp])

        #import pdb; pdb.set_trace()
        if all([isinstance(a,Cast) and a.obj.type == int for a in self.args]):
            # TODO this may be superflous if both args got converted to float in the translation stage -> move toFloat partially to the analysis stage.
            self.result.llvm = builder.fptosi(self.result.llvm, py_type_to_llvm(int), "(int)"+self.result.name)

class BINARY_TRUE_DIVIDE(BinaryOp):
    b_func = {float: 'fdiv'}

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        # The result of `/', true division, is always a float
        self.result.type = float
        super().type_eval(func)

#class InplaceOp(BinaryOp):
#    @pop_stack(2)
#    def stack_eval(self, func, stack):
#        tp = unify_type(self.args[0].type, self.args[1].type, self.debuginfo)
#        if tp != self.args[0].type:
#            self.result = Local.tmp(self.args[0])
#            self.result.type = tp
#        else:
#            self.result = self.args[0]
#        stack.push(self.result)
#
#    def translate(self, module, builder):
#        self.floatArg(builder)
#        f = getattr(builder, self.builderFuncName())
#        self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

# Inplace operators don't have a semantic difference when used on primitive types
class INPLACE_ADD(BINARY_ADD): pass
class INPLACE_SUBTRACT(BINARY_SUBTRACT): pass
class INPLACE_MULTIPLY(BINARY_MULTIPLY): pass
class INPLACE_TRUE_DIVIDE(BINARY_TRUE_DIVIDE): pass
class INPLACE_FLOOR_DIVIDE(BINARY_FLOOR_DIVIDE): pass
class INPLACE_MODULO(BINARY_MODULO): pass

class COMPARE_OP(Bytecode):
    b_func = {float: 'fcmp', int: 'icmp', bool: 'icmp'}
    op = None

    icmp = {'==': llvm.core.ICMP_EQ,
            '!=': llvm.core.ICMP_NE,
            '>':  llvm.core.ICMP_SGT,
            '>=': llvm.core.ICMP_SGE,
            '<':  llvm.core.ICMP_SLT,
            '<=': llvm.core.ICMP_SLE,
            }

    fcmp = {'==': llvm.core.FCMP_OEQ,
            '!=': llvm.core.FCMP_ONE,
            '>':  llvm.core.FCMP_OGT,
            '>=': llvm.core.FCMP_OGE,
            '<':  llvm.core.FCMP_OLT,
            '<=': llvm.core.FCMP_OLE,
            }

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addCmp(self, op):
        self.op = op

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        #func.retype(self.result.unify_type(bool, self.debuginfo))
        self.result.type = bool
        if self.args[0].type != self.args[1].type:
            raise TypingError("Comparing different types ({0} with {1})".format(self.args[0].type, self.args[1].type))

    def translate(self, module, builder):
        # assume both types are the same, see @stack_eval
        tp = self.args[0].type
        if not self.args[0].type in self.b_func:
            raise UnimplementedError(tp)

        f = getattr(builder, self.b_func[tp])
        m = getattr(self,    self.b_func[tp])

        self.result.llvm = f(m[self.op], self.args[0].llvm, self.args[1].llvm, self.result.name)

class RETURN_VALUE(BlockTerminal, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        #import pdb; pdb.set_trace()
        self.result = self.popFirstArg()

    def type_eval(self, func):
        for arg in self.args:
            func.retype(self.result.unify_type(arg.type, self.debuginfo))

    def translate(self, module, builder):
        builder.ret(self.result.llvm)

class HasTarget(object):
    target_label = None
    target_bc = None
    def setTargetBytecode(self, bc):
        #import pdb; pdb.set_trace()
        self.target_bc = bc

    def updateTargetBytecode(self, old_bc, new_bc):
        self.setTargetBytecode(new_bc)

    def setTarget(self, label):
        self.target_label = label

    def __str__(self):
        return "{0} {1} {2}".format(
                    self.__class__.__name__,
                    self.target_label,
                    ", ".join([str(v) for v in self.args]))


class Jump(BlockTerminal, HasTarget, IR):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        return False

    def stack_eval(self, func, stack):
        tos = stack.peek()
        if tos:
            tos.bc = self
        return [(self.target_bc, stack)]

    def type_eval(self,func):
        pass

    def translate(self, module, builder):
        builder.branch(self.target_bc.block)

class Jump_if_X_or_pop(Jump):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        self.fallthrough = self.next
        return True

    def updateTargetBytecode(self, old_bc, new_bc):
        if old_bc == self.target_bc:
            self.setTargetBytecode(new_bc)
        else:
            assert self.fallthrough == old_bc
            self.fallthrough = new_bc

    @pop_stack(1)
    def stack_eval(self, func, stack):
        stack2 = stack.clone()
        r = []
        # if X, push back onto stack and jump:
        self.args[0].bc = self
        stack.push(self.args[0])
        r.append((self.target_bc, stack))
        # else continue with the next instruction (and keep the popped value)
        if not stack2.empty():
            # TODO: is stack2 always empty?
            tos = stack2.peek()
            if tos:
                tos.bc = self
        r.append((self.next, stack2))

        return r

class JUMP_IF_FALSE_OR_POP(Jump_if_X_or_pop, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        #import pdb; pdb.set_trace()
        builder.cbranch(self.args[0].llvm, self.next.block, self.target_bc.block)

class JUMP_IF_TRUE_OR_POP(Jump_if_X_or_pop, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.target_bc.block, self.next.block)

class Pop_jump_if_X(Jump):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        self.fallthrough = self.next
        return True

    def updateTargetBytecode(self, old_bc, new_bc):
        if old_bc == self.target_bc:
            self.setTargetBytecode(new_bc)
        else:
            assert self.fallthrough == old_bc
            self.fallthrough = new_bc

    @pop_stack(1)
    def stack_eval(self, func, stack):
        r = []
        # if X, jump
        self.args[0].bc = self
        r.append((self.target_bc, stack))
        # else continue to the next instruction
        r.append((self.next, stack))
        # (pop happens in any case)

        return r

class POP_JUMP_IF_FALSE(Pop_jump_if_X, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.next.block, self.target_bc.block)

class POP_JUMP_IF_TRUE(Pop_jump_if_X, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.target_bc.block, self.next.block)

class SETUP_LOOP(BlockStart, HasTarget, Bytecode):
    """
    Will either be rewritten (for loop) or has no effect other than mark the
    start of a block (while loop).
    """
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class POP_BLOCK(BlockEnd, Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class LOAD_GLOBAL(Bytecode):
    var = None

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        self.args.append(name)

    def stack_eval(self, func, stack):
        #pdb.set_trace()
        self.var = func.loadGlobal(self.args[0])
        # TODO: remove these isinstance checks and just check for GlobalVariable else return directly?
        if isinstance(self.var, Function):
            self.result = self.var
        elif isinstance(self.var, types.ModuleType):
            self.result = self.var
        elif isinstance(self.var, type):
            self.result = self.var
        elif isinstance(self.var, GlobalVariable):
            self.result = Register(func)
        else:
            raise UnimplementedError("Unknown global type {0}".format(type(self.result)))
        stack.push(self.result)

    def translate(self, module, builder):
        if isinstance(self.var, Function):
            pass
        elif isinstance(self.var, GlobalVariable):
            self.result.llvm = builder.load(self.var.llvm)

    def type_eval(self, func):
        if isinstance(self.var, GlobalVariable):
            self.result.unify_type(self.var.type, self.debuginfo)

class LOAD_ATTR(Bytecode):
    discard = True
    var = None

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        self.args.append(name)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        if isinstance(self.args[1], types.ModuleType):
            self.result = func.module.loadExt(self.args[1], self.args[0])
        else:
            raise UnimplementedError("Cannot load attribute {0} of an object with type {1}".format(self.args[0], type(self.args[1])))
        stack.push(self.result)

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class CALL_FUNCTION(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.intrinsic_class = None
        self.intrinsic = None

    def addRawArg(self, arg):
        self.num_pos_args = arg & 0xFF
        self.num_kw_args = (arg >> 8) & 0xFF

    def separateArgs(self):
        self.func = self.args[0]
        args = self.args[1:]

        #pdb.set_trace()
        assert len(args) == self.num_pos_args + self.num_kw_args*2

        self.kw_args = {}
        for i in range(self.num_kw_args):
            # the list is reversed, so the value comes first
            value = args.pop()
            key = args.pop()
            # key is a Const object, unwrap it
            self.kw_args[key.value] = value

        # remainder is positional
        self.args = args

    def stack_eval(self, func, stack):
        while True:
            arg = stack.pop()
            self.args.append(arg)
            if isinstance(arg, Function):
                break
        self.args.reverse()
        self.separateArgs()

        self.result = Register(func)
        stack.push(self.result)

        self.intrinsic_class = getIntrinsic(self.func)

        if self.intrinsic_class == None:
            func.module.functionCall(self.func, self.args, self.kw_args)

    def type_eval(self, func):
        args = self.func.combineAndCheckArgs(self.args, self.kw_args)

        if self.intrinsic_class:
            if not self.intrinsic:
                self.intrinsic = self.intrinsic_class(args)
            tp = self.intrinsic.getReturnType()
        else:
            tp = self.func.getReturnType()
        tp_change = self.result.unify_type(tp, self.debuginfo)

        if self.result.type == NoType:
            func.impl.analyzeAgain() # redo analysis, right now return type is not known
        else:
            func.retype(tp_change)

    def translate(self, module, builder):
        if self.intrinsic:
            self.result.llvm = self.intrinsic.translate(module, builder)
        else:
            args = self.func.combineAndCheckArgs(self.args, self.kw_args)
            #logging.debug("Call using args: " + str(args))
            #logging.debug("Call using arg_types: " + str(list(map (type, args))))

            self.result.llvm = builder.call(self.func.llvm, [arg.llvm for arg in args])

class GET_ITER(Poison, Bytecode):
    """WIP"""
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class FOR_ITER(Poison, HasTarget, Bytecode):
    """WIP"""
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class JUMP_ABSOLUTE(Jump, Bytecode):
    """WIP"""
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class JUMP_FORWARD(Jump, Bytecode):
    """WIP"""
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)


class ForLoop(IR):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def setLoopVar(self, loop_var):
        self.loop_var = loop_var
    def setLimit(self, limit):
        self.limit = limit
    def setEndLoc(self, end_loc):
        self.end_loc = end_loc
    def setTestLoc(self, loc):
        self.test_loc = loc

    def rewrite(self, func):
        last = self

        # init
        b = LOAD_CONST(func.impl, self.debuginfo)
        b.addArg(Const(0))
        last.insert_after(b)
        last = b

        b = STORE_FAST(func.impl, self.debuginfo)
        #import pdb; pdb.set_trace()
        b.addArg(self.loop_var)
        b.new_allocate = True
        last.insert_after(b)
        last = b

        # test
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        b.loc = self.test_loc
        func.replaceLocation(b)
        last.insert_after(b)
        last = b

        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.limit)
        last.insert_after(b)
        last = b

        b = COMPARE_OP(func.impl, self.debuginfo)
        b.addCmp('>=')
        last.insert_after(b)
        last = b

        b = POP_JUMP_IF_TRUE(func.impl, self.debuginfo)
        b.setTarget(self.end_loc)
        last.insert_after(b)
        last = b


        # $body, keep, find the end of it
        while b.next != None:
            b = b.next
        assert isinstance(b, BlockEnd)
        #import pdb; pdb.set_trace()
        jump_loc = b.loc
        last = b.prev
        b.remove()

        # go back to the JUMP and switch locations
        incr_loc = last.loc
        last.loc = jump_loc
        func.replaceLocation(last)

        # increment
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        b.loc = incr_loc
        func.replaceLocation(b)
        last.insert_before(b)

        b = LOAD_CONST(func.impl, self.debuginfo)
        b.addArg(Const(1))
        last.insert_before(b)

        b = INPLACE_ADD(func.impl, self.debuginfo)
        last.insert_before(b)

        b = STORE_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        last.insert_before(b)

        # JUMP to COMPARE_OP is already part of the bytecodes
        

    def stack_eval(self, func, stack):
        #self.result = func.getOrNewRegister(self.loop_var)
        #stack.push(self.result)
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        #self.result.unify_type(int, self.debuginfo)
        pass

#---

opconst = {}
# Get all contrete subclasses of Bytecode and register them
for name in dir(sys.modules[__name__]):
    obj = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(obj, Bytecode) and len(obj.__abstractmethods__) == 0:
            opconst[dis.opmap[name]] = obj
    except TypeError:
        pass

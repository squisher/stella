import dis
from .llvm import *
from .exc import *
from abc import ABCMeta, abstractmethod, abstractproperty
from llvm.core import INTR_FLOOR

class CastableMixin(object):
    def __init__(self):
        self.orig_type = None

    def toFloat(self, builder):
        #import pdb; pdb.set_trace()
        assert (self.type == int)
        self.llvm = builder.sitofp(self.llvm, py_type_to_llvm(float), "(float)"+self.name)
        self.orig_type = self.type
        self.type = float

class Variable(CastableMixin, object):
    name = None
    type = None

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name + self.type

class Const(object):
    type = None
    value = None

    def __init__(self, value):
        self.value = value
        self.type = type(value)
        self.llvm = llvm_constant(value)
        self.name = str(value)

    def __str__(self):
        return str(self.value)

    def toFloat(self, builder):
        assert (self.type == int)
        self.orig_type = self.type
        self.type = float
        self.value = float(self.value)
        self.llvm = llvm_constant(self.value)


class Local(Variable):
    @staticmethod
    def tmp(template):
        l = Local('')
        if isinstance(template, Variable):
            l.type = template.type
        else:
            l.type = template
        return l

    def __str__(self):
        if self.name:
            return self.name + str(self.type)
        else:
            return "$" + str(self.type)

    def __repr__(self):
        return self.__str__()


def unify_type(tp1, tp2, debuginfo):
    if tp1 == tp2:  return tp1
    if tp1 == None: return tp2
    if tp2 == None: return tp1
    if (tp1 == int and tp2 == float) or (tp1 == float and tp2 == int):
        return float
    raise TypingError ("Unifying of types " + str(tp1) + " and " + str(tp2) + " not yet implemented", debuginfo)

def use_stack(n):
    """
    Decorator, it takes n items off the stack
    and adds them as bytecode arguments.
    """
    def extract_n(f):
        def extract_from_stack(self, *f_args):
            args = []
            for i in range(n):
                args.append(self.stack.pop())
            args.reverse()
            if self.args == None:
                self.args = args
            else:
                self.args.extend(args)
            return f(self, *f_args)
        return extract_from_stack
    return extract_n

class Bytecode(metaclass=ABCMeta):
    args = None
    result = None
    debuginfo = None
    discard = False # true if it should be removed in the register representation
    llvm = None

    def __init__(self, debuginfo, stack):
        self.debuginfo = debuginfo
        self.stack = stack

    def addConst(self, arg):
        self.addArg(Const(arg))

    def addArg(self, arg):
        if self.args == None:
            self.args = []
        self.args.append(arg)

    def floatArg(self, builder):
        if self.result.type == float:
            for arg in self.args:
                if arg.type == int:
                    arg.toFloat(builder)
            return True
        return False

    @abstractmethod
    def eval(self, func):
        pass


class LOAD_FAST(Bytecode):
    discard = True
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self, func):
        self.result = func.locals[self.args[0]]
        self.stack.push(self.result)

class STORE_FAST(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(1)
    def eval(self, func):
        var = Local(self.args[0])
        var.type = self.args[1].type
        func.locals[var.name] = var
        self.args[0] = var
        self.result = var

    def translate(self, module, builder):
        self.result.llvm = self.args[1].llvm

    def __str__(self):
        return "STORE_FAST {0}, {1}".format(self.args[0].name, self.args[1])

class LOAD_CONST(Bytecode):
    discard = True
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self, func):
        self.result = self.args[0]
        self.stack.push(self.result)

class BinaryOp(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(2)
    def eval(self, func):
        self.result = Local.tmp(unify_type(self.args[0].type, self.args[1].type, self.debuginfo))
        self.stack.push(self.result)

    def builderFuncName(self):
        try:
            return self.b_func[self.result.type]
        except KeyError:
            raise TypingError("{0} does not yet implement type {1}".format(self.__class__.__name__, self.result.type))

    def translate(self, module, builder):
        self.floatArg(builder)
        f = getattr(builder, self.builderFuncName())
        self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

    def __str__(self):
        return '{0} {1}, {2}'.format(self.__class__.__name__, *self.args)

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
    b_func = {float: INTR_POW, int: INTR_POWI}

    @use_stack(2)
    def eval(self, func):
        # TODO if args[1] is int but negative, then the result will be float, too!
        if self.args[0].type == int and self.args[1].type == int:
            self.result = Local.tmp(int)
        else:
            self.result = Local.tmp(float)
        self.stack.push(self.result)

    def translate(self, module, builder):
        # llvm.pow[i]'s first argument always has to be float
        if self.args[0].type == int:
            self.args[0].toFloat(builder)

        if self.args[1].type == int:
            # powi takes a i32 argument
            power = builder.trunc(self.args[1].llvm, tp_int32, '(i32)'+self.args[1].name)
        else:
            power = self.args[1].llvm

        llvm_pow = Function.intrinsic(module, self.b_func[self.args[1].type], [py_type_to_llvm(self.args[0].type)])
        pow_result = builder.call(llvm_pow, [self.args[0].llvm, power])

        if self.args[0].orig_type == int and self.args[1].type == int:
            # cast back to an integer
            self.result.llvm = builder.fptosi(pow_result, py_type_to_llvm(int))
        else:
            self.result.llvm = pow_result

#class BINARY_FLOOR_DIVIDE(BinaryOp):
#    """Floor divide for float, C integer divide for ints. Fast, but unlike the Python semantics for integers"""
#    b_func = {float: 'fdiv', int: 'sdiv'}
#
#    def translate(self, module, builder):
#        self.floatArg(builder)
#        f = getattr(builder, self.builderFuncName())
#        if self.result.type == float:
#            tmp = f(self.args[0].llvm, self.args[1].llvm, self.result.name)
#            llvm_floor = Function.intrinsic(module, INTR_FLOOR, [py_type_to_llvm(self.result.type)])
#            self.result.llvm = builder.call(llvm_floor, [tmp])
#        else:
#            self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

class BINARY_FLOOR_DIVIDE(BinaryOp):
    """Python compliant `//' operator: Slow since it has to perform type conversions and floating point division for integers"""
    b_func = {float: 'fdiv', int: 'fdiv'} # NOT USED, but required to make it a concrete class

    def translate(self, module, builder):
        # convert all arguments to float, since fp division is required to apply floor
        for arg in self.args:
            if arg.type == int:
                arg.toFloat(builder)

        tmp = builder.fdiv(self.args[0].llvm, self.args[1].llvm, self.result.name)
        llvm_floor = Function.intrinsic(module, INTR_FLOOR, [py_type_to_llvm(float)])
        self.result.llvm = builder.call(llvm_floor, [tmp])

        if self.result.type == int:
            # TODO this may be superflous if both args got converted to float in the translation stage -> move toFloat partially to the analysis stage.
            self.result.llvm = builder.fptosi(self.result.llvm, py_type_to_llvm(int), "(int)"+self.result.name)

class BINARY_TRUE_DIVIDE(BinaryOp):
    b_func = {float: 'fdiv'}

    @use_stack(2)
    def eval(self, func):
        # The result of `/', true division, is always a float
        self.result = Local.tmp(float)
        self.stack.push(self.result)

#class InplaceOp(BinaryOp):
#    @use_stack(2)
#    def eval(self, func):
#        tp = unify_type(self.args[0].type, self.args[1].type, self.debuginfo)
#        if tp != self.args[0].type:
#            self.result = Local.tmp(self.args[0])
#            self.result.type = tp
#        else:
#            self.result = self.args[0]
#        self.stack.push(self.result)
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

    icmp = {'==': ICMP_EQ,
            '!=': ICMP_NE,
            '>':  ICMP_SGT,
            '>=': ICMP_SGE,
            '<':  ICMP_SLT,
            '<=': ICMP_SLE,
            }

    fcmp = {'==': FCMP_OEQ,
            '!=': FCMP_ONE,
            '>':  FCMP_OGT,
            '>=': FCMP_OGE,
            '<':  FCMP_OLT,
            '<=': FCMP_OLE,
            }

    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def addCmp(self, op):
        self.op = op

    @use_stack(2)
    def eval(self, func):
        self.result = Local.tmp(bool)
        self.stack.push(self.result)
        if self.args[0].type != self.args[1].type:
            raise TypingError("Comparing different types ({0} with {1})".format(self.args[0].type, self.args[1].type))

    def translate(self, module, builder):
        # assume both types are the same, see @eval
        tp = self.args[0].type
        if not self.args[0].type in self.b_func:
            raise UnimplementedError(tp)

        f = getattr(builder, self.b_func[tp])
        m = getattr(self,    self.b_func[tp])

        self.result.llvm = f(m[self.op], self.args[0].llvm, self.args[1].llvm, self.result.name)

class RETURN_VALUE(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(1)
    def eval(self, func):
        pass

    def translate(self, module, builder):
        builder.ret(self.args[0].llvm)

    def __str__(self):
        return 'RETURN ' + str(self.args[0])

opconst = {}
# Get all contrete subclasses of Bytecode and register them
for name in dir(sys.modules[__name__]):
    obj = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(obj, Bytecode) and len(obj.__abstractmethods__) == 0:
            opconst[dis.opmap[name]] = obj
    except TypeError:
        pass

import dis
from stella.llvm import *
from abc import ABCMeta, abstractmethod, abstractproperty
from llvm.core import INTR_FLOOR

class Variable(object):
    name = None
    type = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name + self.type

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
    and adds the as arguments to the bytecode.
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

    def __init__(self, debuginfo, stack):
        self.debuginfo = debuginfo
        self.stack = stack

    def addArg(self, arg):
        if self.args == None:
            self.args = []
        self.args.append(arg)

    def floatArg(self, builder):
        if self.result.type == float:
            for arg in self.args:
                if arg.type == int:
                    arg.llvm = builder.sitofp(arg.llvm, py_type_to_llvm(float), "(float)"+arg.name)
            return True
        return False

    @abstractmethod
    def eval(self):
        pass


class LOAD_FAST(Bytecode):
    discard = True
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        self.result = self.args[0]
        self.stack.push(self.result)

class BinaryOp(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(2)
    def eval(self):
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
                arg.llvm = builder.sitofp(arg.llvm, py_type_to_llvm(float), "(float)"+arg.name)

        tmp = builder.fdiv(self.args[0].llvm, self.args[1].llvm, self.result.name)
        llvm_floor = Function.intrinsic(module, INTR_FLOOR, [py_type_to_llvm(float)])
        self.result.llvm = builder.call(llvm_floor, [tmp])

        if self.result.type == int:
            self.result.llvm = builder.fptosi(self.result.llvm, py_type_to_llvm(int), "(int)"+self.result.name)

class BINARY_TRUE_DIVIDE(BinaryOp):
    b_func = {float: 'fdiv'}

    @use_stack(2)
    def eval(self):
        # The result of `/', true division, is always a float
        self.result = Local.tmp(float)
        self.stack.push(self.result)

class RETURN_VALUE(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(1)
    def eval(self):
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

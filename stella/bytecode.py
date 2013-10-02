import dis
from stella.llvm import *

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

class Bytecode(object):
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

class LOAD_FAST(Bytecode):
    discard = True
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        self.result = self.args[0]
        self.stack.push(self.result)

class BINARY_ADD(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(2)
    def eval(self):
        self.result = Local.tmp(unify_type(self.args[0].type, self.args[1].type, self.debuginfo))
        self.stack.push(self.result)

    def translate(self, builder):
        fadd = any([arg.type == float for arg in self.args])
        import pdb
        #pdb.set_trace()
        if fadd:
            for arg in self.args:
                if arg.type == int:
                    arg.llvm = builder.sitofp(arg.llvm, py_type_to_llvm(float), "(float)"+arg.name)

            self.result.llvm = builder.fadd(self.args[0].llvm, self.args[1].llvm, self.result.name)
        else:
            self.result.llvm = builder.add(self.args[0].llvm, self.args[1].llvm, self.result.name)

    def __str__(self):
        return 'ADD {0}, {1}'.format(*self.args)

class BINARY_SUBTRACT(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(2)
    def eval(self):
        self.result = Local.tmp(unify_type(self.args[0].type, self.args[1].type, self.debuginfo))
        self.stack.push(self.result)

    def translate(self, builder):
        self.result.llvm = builder.sub(self.args[0].llvm, self.args[1].llvm, self.result.name)

    def __str__(self):
        return 'SUB {0}, {1}'.format(*self.args)

class RETURN_VALUE(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    @use_stack(1)
    def eval(self):
        pass

    def translate(self, builder):
        builder.ret(self.args[0].llvm)

    def __str__(self):
        return 'RETURN ' + str(self.args[0])

opconst = {}
opconst[dis.opmap['LOAD_FAST']] = LOAD_FAST
opconst[dis.opmap['BINARY_ADD']] = BINARY_ADD
opconst[dis.opmap['BINARY_SUBTRACT']] = BINARY_SUBTRACT
opconst[dis.opmap['RETURN_VALUE']] = RETURN_VALUE


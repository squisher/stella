import dis

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

    def eval(self):
        arg1 = self.stack.pop()
        self.addArg(arg1)
        arg2 = self.stack.pop()
        self.addArg(arg2)
        self.result = Local.tmp(unify_type(arg1.type, arg2.type, self.debuginfo))
        self.stack.push(self.result)

    def translate(self, builder):
        self.result.llvm = builder.add(self.args[0].llvm, self.args[1].llvm, self.result.name)

class BINARY_SUBTRACT(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        arg1 = self.stack.pop()
        self.addArg(arg1)
        arg2 = self.stack.pop()
        self.addArg(arg2)
        self.result = Local.tmp(unify_type(arg1.type, arg2.type, self.debuginfo))
        self.stack.push(self.result)

    def translate(self, builder):
        self.result.llvm = builder.sub(self.args[0].llvm, self.args[1].llvm, self.result.name)

class RETURN_VALUE(Bytecode):
    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        self.addArg(self.stack.pop())

    def translate(self, builder):
        builder.ret(self.args[0].llvm)

    def __str__(self):
        return 'RETURN ' + str(self.args[0])

opconst = {}
opconst[dis.opmap['LOAD_FAST']] = LOAD_FAST
opconst[dis.opmap['BINARY_ADD']] = BINARY_ADD
opconst[dis.opmap['BINARY_SUBTRACT']] = BINARY_SUBTRACT
opconst[dis.opmap['RETURN_VALUE']] = RETURN_VALUE


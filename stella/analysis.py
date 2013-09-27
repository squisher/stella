import dis
import inspect

import logging

class Debuginfo(object):
    line = None
    filename = None

class Stack(object):
    backend = []
    def push(self, item):
        logging.debug("[Stack] Pushing " + str(item))
        self.backend.append(item)
    def pop(self):
        item = self.backend.pop()
        logging.debug("[Stack] Popping " + str(item))
        return item

opconst = {}
class Bytecode(object):
    args = []
    result = None
    name = None
    line = 0

    def __init__(self, line, stack):
        self.line = line
        self.stack = Stack()

    def addArg(self, arg):
        self.args.append(arg)

class LOAD_FAST(Bytecode):
    name = 'LOAD_FAST'

    def __init__(self, line, stack):
        super().__init__(line, stack)

    def eval(self):
        self.result = self.args[0]
        self.stack.push(self.result)

class BINARY_ADD(Bytecode):
    name = 'BINARY_ADD'

    def __init__(self, line, stack):
        super().__init__(line, stack)

    def eval(self):
        tp1 = self.stack.pop()
        tp2 = self.stack.pop()
        if tp1.type != tp2.type:
            raise TypingError(self.line, str(tp1.type) + " != " + str(tp2.type))
        self.result = tp1
        self.stack.push(Local.tmp(tp1))

class RETURN_VALUE(Bytecode):
    name = 'RETURN_VALUE'

    def __init__(self, line, stack):
        super().__init__(line, stack)

    def eval(self):
        self.result = self.stack.pop()

    def __str__(self):
        return 'RETURN ' + str(self.result)

opconst[dis.opmap['LOAD_FAST']] = LOAD_FAST
opconst[dis.opmap['BINARY_ADD']] = BINARY_ADD
opconst[dis.opmap['RETURN_VALUE']] = RETURN_VALUE

class UnsupportedOpcode(Exception):
    def __init__(self, op, line):
        super().__init__("Unsupported opcode {0} on line {1}".format(dis.opname[op], line))

class TypingError(TypeError):
    def __init__(self, line, msg):
        super().__init__(line + ': ' + msg)

class Function(object):
    f = None
    locals = dict()
    args = []
    return_tp = None
    stack = []
    bytecodes = []

    def __init__(self, f):
        self.f = f
        argspec = inspect.getargspec(f)
        self.args = [Local(n) for n in argspec.args]
        for arg in self.args:
            self.locals[arg.name] = arg

    def analyze(self, *args):
        for i in range(len(args)):
            self.args[i].type = type(args[i])
        logging.debug("Analysis of " + str(self.f) + "(" + str(self.args) + ")")
        self.disassemble()
        logging.debug("last bytecode: " + str(self.bytecodes[-1]))

    def disassemble(self):
        """Disassemble a code object."""
        lasti=-1
        co = self.f.__code__
        code = co.co_code
        labels = dis.findlabels(code)
        linestarts = dict(dis.findlinestarts(co))
        n = len(code)
        i = 0
        extended_arg = 0
        free = None
        line = 0
        while i < n:
            op = code[i]
            if i in linestarts:
                line = linestarts[i]

            if extended_arg == 0 and op in opconst:
                bc = opconst[op](line, self.stack)
            else:
                raise UnsupportedOpcode(op, line)

            #print(repr(i).rjust(4), end=' ')
            #print(dis.opname[op].ljust(20), end=' ')
            i = i+1
            if op >= dis.HAVE_ARGUMENT:
                oparg = code[i] + code[i+1]*256 + extended_arg
                extended_arg = 0
                i = i+2
                if op == dis.EXTENDED_ARG:
                    extended_arg = oparg*65536
                #print(repr(oparg).rjust(5), end=' ')
                if op in dis.hasconst:
                    #print('(' + repr(co.co_consts[oparg]) + ')', end=' ')
                    bc.addConst(co.co_consts[oparg])
                elif op in dis.hasname:
                    print('(' + co.co_names[oparg] + ')', end=' ')
                elif op in dis.hasjrel:
                    print('(to ' + repr(i + oparg) + ')', end=' ')
                elif op in dis.haslocal:
                    #print('(' + co.co_varnames[oparg] + ')', end=' ')
                    bc.addArg(self.locals[co.co_varnames[oparg]])
                elif op in dis.hascompare:
                    print('(' + cmp_op[oparg] + ')', end=' ')
                elif op in dis.hasfree:
                    if free is None:
                        free = co.co_cellvars + co.co_freevars
                    print('(' + free[oparg] + ')', end=' ')

            bc.eval()
            self.bytecodes.append(bc)


class Variable(object):
    name = None
    type = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name + ':' + self.type

class Local(Variable):
    @staticmethod
    def tmp(template):
        l = Local('')
        l.type = template.type
        return l

    def __str__(self):
        if self.name:
            return self.name + ":" + str(self.type)
        else:
            return "<L>:" + str(self.type)

    def __repr__(self):
        return self.__str__()


def main(f, *args):
    f = Function(f)
    f.analyze(*args)

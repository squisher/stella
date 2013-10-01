import dis
import inspect

import logging

class DebugInfo(object):
    line = None
    filename = None
    def __init__(self, filename, line):
        self.line = line
        self.filename = filename
    def __str__(self):
        return self.filename + ':' + str(self.line)

class Stack(object):
    backend = None

    def __init__(self):
        self.backend = []

    def push(self, item):
        logging.debug("[Stack] Pushing " + str(item))
        self.backend.append(item)
    def pop(self):
        item = self.backend.pop()
        logging.debug("[Stack] Popping " + str(item))
        return item

class Bytecode(object):
    args = None
    result = None
    name = None
    debuginfo = None

    def __init__(self, debuginfo, stack):
        self.debuginfo = debuginfo
        self.stack = stack

    def addArg(self, arg):
        if self.args == None:
            self.args = []
        self.args.append(arg)

class LOAD_FAST(Bytecode):
    name = 'LOAD_FAST'

    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        self.result = self.args[0]
        self.stack.push(self.result)

class BINARY_ADD(Bytecode):
    name = 'BINARY_ADD'

    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        tp1 = self.stack.pop()
        tp2 = self.stack.pop()
        if tp1.type != tp2.type:
            raise TypingError(self.debuginfo, str(tp1.type) + " != " + str(tp2.type))
        self.result = tp1
        self.stack.push(Local.tmp(tp1))

class RETURN_VALUE(Bytecode):
    name = 'RETURN_VALUE'

    def __init__(self, debuginfo, stack):
        super().__init__(debuginfo, stack)

    def eval(self):
        self.addArg(self.stack.pop())

    def __str__(self):
        return 'RETURN ' + str(self.args[0])

opconst = {}
opconst[dis.opmap['LOAD_FAST']] = LOAD_FAST
opconst[dis.opmap['BINARY_ADD']] = BINARY_ADD
opconst[dis.opmap['RETURN_VALUE']] = RETURN_VALUE


class BaseException(Exception):
    def __init__(self, msg, debuginfo):
        super().__init__('{0} at {1}'.format(msg, debuginfo))

class UnsupportedOpcode(BaseException):
    def __init__(self, op, debuginfo):
        super().__init__(dis.opname[op], debuginfo)

class TypingError(BaseException):
    pass

def unify_type(tp1, tp2):
    if tp1 == tp2:  return tp1
    if tp1 == None: return tp2
    if tp2 == None: return tp1
    raise TypingError ("Unifying of types " + str(tp1) + " and " + str(tp2) + " not yet implemented")

class Function(object):
    f = None
    locals = dict()
    args = []
    return_tp = None
    stack = Stack()
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
        for bc in self.bytecodes:
            if isinstance(bc, RETURN_VALUE):
                self.return_tp = unify_type(self.return_tp, bc.args[0].type)
        logging.debug("last bytecode: " + str(self.bytecodes[-1]))
        logging.debug("returning type " + str(self.return_tp))

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

            di = DebugInfo(co.co_filename, line)

            if extended_arg == 0 and op in opconst:
                bc = opconst[op](di, self.stack)
            else:
                raise UnsupportedOpcode(op, di)

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
        return self.name + self.type

class Local(Variable):
    @staticmethod
    def tmp(template):
        l = Local('')
        l.type = template.type
        return l

    def __str__(self):
        if self.name:
            return self.name + str(self.type)
        else:
            return "$" + str(self.type)

    def __repr__(self):
        return self.__str__()


def main(f, *args):
    f = Function(f)
    f.analyze(*args)

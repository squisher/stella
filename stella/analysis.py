import dis
import inspect

import logging

from .bytecode import *
from .exc import *

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


class Function(object):
    def __init__(self, f):
        self.locals = dict()
        self.return_tp = None
        self.stack = Stack()
        self.bytecodes = []

        self.f = f
        argspec = inspect.getargspec(f)
        self.args = [Local(n) for n in argspec.args]
        for arg in self.args:
            self.locals[arg.name] = arg

    def getName(self):
        return self.f.__name__

    def analyze(self, *args):
        for i in range(len(args)):
            self.args[i].type = type(args[i])
        logging.debug("Analysis of " + str(self.f) + "(" + str(self.args) + ")")
        self.disassemble()
        for bc in self.bytecodes:
            if isinstance(bc, RETURN_VALUE):
                self.return_tp = unify_type(self.return_tp, bc.args[0].type, bc.debuginfo)
        #logging.debug("last bytecode: " + str(self.bytecodes[-1]))
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
            try:
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
                        bc.addArg(co.co_varnames[oparg])
                    elif op in dis.hascompare:
                        #print('(' + dis.cmp_op[oparg] + ')', end=' ')
                        bc.addCmp(dis.cmp_op[oparg])
                    elif op in dis.hasfree:
                        if free is None:
                            free = co.co_cellvars + co.co_freevars
                        print('(' + free[oparg] + ')', end=' ')

                bc.eval(self)
            except StellaException as e:
                e.addDebug(di)
                raise
            if not bc.discard:
                self.bytecodes.append(bc)


def main(f, *args):
    f = Function(f)
    f.analyze(*args)
    return f

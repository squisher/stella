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

    def __init__(self, name="Stack"):
        self.backend = []
        self.name = name

    def push(self, item):
        logging.debug("["+self.name+"] Pushing " + str(item))
        self.backend.append(item)
    def pop(self):
        item = self.backend.pop()
        logging.debug("["+self.name+"] Popping " + str(item))
        return item
    def empty(self):
        return len(self.backend) == 0

class Function(object):
    def __init__(self, f):
        self.locals = dict()
        self.result = Variable('__return__') # TODO possible name conflicts?
        self.stack = Stack()
        self.bytecodes = None # pointer to the first bytecode
        self.blocks = {}
        self.labels = {}
        self.todo = Stack("Todo")

        self.f = f
        argspec = inspect.getargspec(f)
        self.args = [Local(n) for n in argspec.args]
        for arg in self.args:
            self.locals[arg.name] = arg

    def getName(self):
        return self.f.__name__

    def getLocal(self, name):
        if name not in self.locals:
            var = Local(name)
            self.locals[name] = var
        return self.locals[name]

    def retype(self, go):
        if go:
            self.analyze_again = True

    def analyze(self, *args):
        for i in range(len(args)):
            self.args[i].type = type(args[i])
        logging.debug("Analysis of " + str(self.f) + "(" + str(self.args) + ")")

        self.disassemble()

        self.todo.push(self.bytecodes)

        while not self.todo.empty():
            self.analyze_again = False
            start = self.todo.pop()
            for bc in start:
                bc.type_eval(self)
                logging.debug("TYPE'D " + str(bc))
                if isinstance(bc, RETURN_VALUE):
                    self.retype(self.result.unify_type(bc.result.type, bc.debuginfo))
            if self.analyze_again:
                self.todo.push(start)

        #logging.debug("last bytecode: " + str(self.bytecodes[-1]))
        logging.debug("returning type " + str(self.result.type))

        logging.debug("PyStack bytecode:")
        for bc in self.bytecodes:
            logging.debug(str(bc))

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
        last_bc = None
        while i < n:
            op = code[i]
            if i in linestarts:
                line = linestarts[i]

            di = DebugInfo(co.co_filename, line)

            if extended_arg == 0 and op in opconst:
                bc = opconst[op](di, self.stack)
            else:
                raise UnsupportedOpcode(op, di)

            if i in labels:
                self.labels[i] = bc

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
                        bc.addArg(self.getLocal(co.co_varnames[oparg]))
                    elif op in dis.hascompare:
                        #print('(' + dis.cmp_op[oparg] + ')', end=' ')
                        bc.addCmp(dis.cmp_op[oparg])
                    elif op in dis.hasfree:
                        if free is None:
                            free = co.co_cellvars + co.co_freevars
                        print('(' + free[oparg] + ')', end=' ')

                bc.stack_eval(self)
                logging.debug("EVAL'D " + str(bc))
            except StellaException as e:
                e.addDebug(di)
                raise

            if not bc.discard:
                if last_bc != None:
                    last_bc.next = bc
                else:
                    self.bytecodes = bc
                last_bc = bc


def main(f, *args):
    f = Function(f)
    f.analyze(*args)
    return f

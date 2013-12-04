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
    def peek(self):
        return self.backend[-1]
    def empty(self):
        return len(self.backend) == 0
    def clone(self):
        s = Stack(self.name)
        s.backend = [x for x in self.backend]
        return s

class Function(object):
    def __init__(self, f):
        self.locals = dict()
        self.result = Variable('__return__') # TODO possible name conflicts?
        self.stack = Stack()
        self.bytecodes = None # pointer to the first bytecode
        self.blocks = {}
        self.labels = {}
        self.todo = Stack("Todo")
        self.incoming_jumps = {}

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

    def retype(self, go = True):
        if go:
            #import pdb; pdb.set_trace()
            self.analyze_again = True

    def analyze(self, *args):
        for i in range(len(args)):
            self.args[i].type = type(args[i])
        logging.debug("Analysis of " + str(self.f) + "(" + str(self.args) + ")")

        logging.debug("Disassembling and Stack->Register conversion")
        self.disassemble()

        self.todo.push(self.bytecodes)

        i = 0
        while not self.todo.empty():
            logging.debug("Type analysis iteration {0}".format(i))
            self.analyze_again = False
            bc_list = self.todo.pop()

            for bc in bc_list:
                bc.type_eval(self)
                logging.debug("TYPE'D " + str(bc))
                if isinstance(bc, RETURN_VALUE):
                    self.retype(self.result.unify_type(bc.result.type, bc.debuginfo))

            if self.analyze_again:
                self.todo.push(bc_list)

            if i > 10:
                raise Exception("Stopping after {0} type analysis iterations (failsafe)".format(i))
            i += 1

        #logging.debug("last bytecode: " + str(self.bytecodes[-1]))
        logging.debug("returning type " + str(self.result.type))

        logging.debug("PyStack bytecode:")
        #import pdb; pdb.set_trace()
        for bc in self.bytecodes:
            logging.debug(str(bc))

    def incoming_jump(self, jump):
        """
        type(jump.args[0]) == Target
        """
        loc = jump.target_label
        if loc in self.incoming_jumps:
            self.incoming_jumps[loc].append(jump)
        else:
            self.incoming_jumps[loc] = [jump]

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
            #import pdb; pdb.set_trace()
            bc.loc = i

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

                    if op in dis.hasconst:
                        #print('(' + repr(co.co_consts[oparg]) + ')', end=' ')
                        bc.addConst(co.co_consts[oparg])
                    elif op in dis.hasname:
                        #print('(' + co.co_names[oparg] + ')', end=' ')
                        raise UnimplementedError('hasname')
                    elif op in dis.hasjrel:
                        #print('(to ' + repr(i + oparg) + ')', end=' ')
                        bc.addTarget(i+oparg)
                    elif op in dis.hasjabs:
                        #print(repr(oparg).rjust(5), end=' ')
                        bc.addTarget(oparg)
                    elif op in dis.haslocal:
                        #print('(' + co.co_varnames[oparg] + ')', end=' ')
                        bc.addArg(self.getLocal(co.co_varnames[oparg]))
                    elif op in dis.hascompare:
                        #print('(' + dis.cmp_op[oparg] + ')', end=' ')
                        bc.addCmp(dis.cmp_op[oparg])
                    elif op in dis.hasfree:
                        if free is None:
                            free = co.co_cellvars + co.co_freevars
                        #print('(' + free[oparg] + ')', end=' ')
                        raise UnimplementedError('hasfree')

                if op in dis.hasjabs or op in dis.hasjrel:
                    self.incoming_jump(bc)

                if bc.loc in self.incoming_jumps:
                    bc_ = PhiNode(di, self.stack)

                    # loc is only used for jumps, and we want to use
                    # bc_ instead of bc as the target, so delete bc's loc
                    # TODO verify that this is not causing issues elsewhere
                    bc_.loc = bc.loc
                    bc.loc = None

                    if not isinstance(bc.prev, Jump):
                        bc__ = Jump(di, self.stack)
                        bc__.addTarget(bc_.loc)
                        self.incoming_jump(bc__)
                        bc__.stack_eval(self)
                        bc__.prev = last_bc
                        last_bc = bc__

                    bc_.stack_eval(self)
                    for i_bc in self.incoming_jumps[bc.loc]:
                        bc_.addArg(i_bc.args[-1])
                        i_bc.addTargetBytecode(bc_)

                    if last_bc != None:
                        last_bc.next = bc_
                        bc_.prev = last_bc
                        last_bc = bc_
                    else:
                        self.bytecodes = bc_
                    last_bc = bc_

                bc.stack_eval(self)

                logging.debug("EVAL'D " + str(bc))
            except StellaException as e:
                e.addDebug(di)
                raise

            if not bc.discard:
                if last_bc != None:
                    last_bc.next = bc
                    bc.prev = last_bc
                else:
                    self.bytecodes = bc
                last_bc = bc


def main(f, *args):
    f = Function(f)
    f.analyze(*args)
    return f

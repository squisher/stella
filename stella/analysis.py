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
    def __str__(self):
        return "["+self.name+"("+str(len(self.backend))+")]"
    def __repr__(self):
        return "["+self.name+"="+", ".join([str(x) for x in self.backend])+"]"
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
        self.bytecodes = None # pointer to the first bytecode
        self.labels = {}
        self.todo = Stack("Todo")
        self.incoming_jumps = {}
        self.fellthrough = False

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

    def intraflow(self):
        for bc in self.bytecodes:
            if isinstance(bc, Jump):
                if bc.processFallThrough():
                    self.incoming_jumps[bc.next].append(bc)
                target_bc = self.labels[bc.target_label]
                bc.addTargetBytecode(target_bc)
                self.incoming_jumps[target_bc].append(bc)

        for bc in self.bytecodes:
            if len(self.incoming_jumps[bc]) > 1:
                if not isinstance(bc.prev, BlockTerminal):
                    #logging.debug("PREV_TYPE " + str(type(bc.prev)))
                    bc_ = Jump(bc.debuginfo)
                    bc_.addTargetBytecode(bc)
                    bc_.addTarget(bc.loc) # for printing purposes only
                    self.insert_before(bc_)

                    logging.debug("IF ADD  " + bc_.locStr())

                bc_ = PhiNode(bc.debuginfo)
                bc_.loc = bc.loc # for printing purposes only

                self.insert_before(bc_)
                logging.debug("IF ADD  " + bc_.locStr())
                #import pdb; pdb.set_trace()


    def analyze(self, *args):
        for i in range(len(args)):
            self.args[i].type = type(args[i])
        logging.debug("Analysis of " + str(self.f) + "(" + str(self.args) + ")")

        logging.debug("Disassembling")
        self.disassemble()

        logging.debug("Building Intra-Flowgraph")
        self.intraflow()

        logging.debug("Stack->Register Conversion")
        stack = Stack()
        self.todo.push((self.bytecodes, stack))

        while not self.todo.empty():
            (bc, stack) = self.todo.pop()
            r = bc.stack_eval(self, stack)
            if r == None:
                # default case: no control flow diversion, just continue with the next
                # instruction in the list
                # Note: the `and not' part is a basic form of dead code elimination
                #       This is used to drop unreachable "return None" which are implicitly added
                #       by Python to the end of functions.
                #       TODO is this the proper way to handle those returns? Any side effects?
                #            NEEDS REVIEW
                #       See also codegen.Program.__init__
                if bc.next and not isinstance(bc, BlockTerminal):
                    self.todo.push((bc.next, stack))
                else:
                    logging.debug("Reached EOP.")
                    assert stack.empty()
            else:
                # there is (one or more) control flow changes, add them all to the todo list
                assert isinstance(r, list)
                for (bc_, stack_) in r:
                    self.todo.push((bc_, stack_))

        logging.debug("Type Analysis")
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

        #logging.debug("PyStack bytecode:")
        #import pdb; pdb.set_trace()
        #for bc in self.bytecodes:
        #    logging.debug(str(bc))

    def insert_after(self, bc):
        if self.last_bc != None:
            self.last_bc.next = bc
            bc.prev = self.last_bc
        else:
            self.bytecodes = bc
        self.last_bc = bc

    def insert_before(self, bc):
        if self.last_bc == None:
            logging.warn("Insert_before called on an empty list")
            insert_after(bc)
        pp = self.last_bc.prev
        pp.next = bc
        self.last_bc.prev = bc

        bc.prev = pp
        bc.next = self.last_bc

    def remove(self, bc):
        if bc in self.incoming_jumps:
            for i_bc in self.incoming_jumps[bc]:
                i_bc.addTargetBytecode(bc.next)

        if bc.prev:
            bc.prev.next = bc.next
        else:
            # it's the first
            self.bytecodes = bc.next
        if bc.next:
            # TODO this should never happen, should it?
            bc.next.prev = bc.prev
        # Note that bc's prev and next remain untouched so that
        # the iterator remains valid

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
        self.last_bc = None
        while i < n:
            op = code[i]
            if i in linestarts:
                line = linestarts[i]

            di = DebugInfo(co.co_filename, line)

            if extended_arg == 0 and op in opconst:
                bc = opconst[op](di)
            else:
                raise UnsupportedOpcode(op, di)
            #import pdb; pdb.set_trace()
            bc.loc = i
            self.incoming_jumps[bc] = []
            self.insert_after(bc)

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

                logging.debug("DIS'D {0}".format(bc.locStr()))
            except StellaException as e:
                e.addDebug(di)
                raise


def main(f, *args):
    f = Function(f)
    f.analyze(*args)
    return f

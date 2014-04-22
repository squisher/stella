import dis
import inspect

import logging

from .bytecode import *
from .exc import *
from . import bytecode

class DebugInfo(object):
    line = None
    filename = None
    def __init__(self, filename, line):
        self.line = line
        self.filename = filename
    def __str__(self):
        return self.filename + ':' + str(self.line)

class Function(object):
    def __init__(self, f, module):
        self.bytecodes = None # pointer to the first bytecode
        self.labels = {}
        self.todo = Stack("Todo")
        self.incoming_jumps = {}

        self.f = f
        self.impl = bytecode.Function(self.getName(), inspect.getargspec(f), module)
        self.module = module

    def getName(self):
        return self.f.__name__
    def __str__(self):
        return self.getName()

    def retype(self, go = True):
        if go:
            #import pdb; pdb.set_trace()
            self.analyze_again = True

    def add_incoming_jump(self, target_bc, source_bc):
        if target_bc in self.incoming_jumps:
            self.incoming_jumps[target_bc].append(source_bc)
        else:
            self.incoming_jumps[target_bc] = [source_bc]

    def replaceLocation(self, bc):
        """Assume that bc.loc points to the new location already."""
        self.labels[bc.loc] = bc

    def rewrite(self):
        self.bytecodes.printAll()
        logging.debug("Rewriting (peephole optimizations) ------------------------------")
        for bc in self.bytecodes:
            if isinstance(bc, FOR_ITER):
                cur = bc.prev
                if not isinstance(cur, GET_ITER):
                    raise UnimplementedError('unsupported for loop')
                cur.remove()
                cur = bc.prev
                if not isinstance(cur, CALL_FUNCTION):
                    raise UnimplementedError('unsupported for loop')
                cur.remove()
                cur = bc.prev
                if not isinstance(cur, LOAD_FAST):
                    raise UnimplementedError('unsupported for loop')
                limit = cur.args[0]
                cur.remove()
                cur = bc.prev
                if not isinstance(cur, LOAD_GLOBAL):
                    raise UnimplementedError('unsupported for loop')
                cur.remove()
                cur = bc.prev
                if not isinstance(cur, SETUP_LOOP):
                    raise UnimplementedError('unsupported for loop')
                end_loc = cur.target_label
                #import pdb; pdb.set_trace()

                for_loop = ForLoop(self, bc.debuginfo)
                for_loop.loc = cur.loc
                # TODO set location for for_loop and transfer jumps!
                for_loop.setLimit(limit)
                for_loop.setEndLoc(end_loc)
                for_loop.setTestLoc(bc.loc)

                cur.insert_after(for_loop)
                cur.remove()

                cur = bc.next
                if not isinstance(cur, STORE_FAST):
                    raise UnimplementedError('unsupported for loop')
                loop_var = cur.args[0]
                for_loop.setLoopVar(loop_var)
                cur.remove()

                bc.remove()
                for_loop.rewrite(self)

        self.bytecodes.printAll()


    def intraflow(self):
        logging.debug("Building Intra-Flowgraph ------------------------------")
        for bc in self.bytecodes:
            if isinstance(bc, Jump):
                if bc.processFallThrough():
                    self.add_incoming_jump(bc.linearNext(), bc)
                target_bc = self.labels[bc.target_label]
                bc.setTargetBytecode(target_bc)
                self.add_incoming_jump(target_bc, bc)

        for bc in self.bytecodes:
            if bc in self.incoming_jumps:
                if not isinstance(bc.linearPrev(), BlockTerminal):
                    #import pdb; pdb.set_trace()
                    #logging.debug("PREV_TYPE " + str(type(bc.prev)))
                    bc_ = Jump(self, bc.debuginfo)
                    bc_.loc = ''
                    bc_.setTargetBytecode(bc)
                    bc_.setTarget(bc.loc) # for printing purposes only
                    bc.insert_before(bc_)
                    self.add_incoming_jump(bc, bc_)

                    logging.debug("IF ADD  " + bc_.locStr())

                if len(self.incoming_jumps[bc]) > 1:
                    bc_ = PhiNode(self.impl, bc.debuginfo)
                    bc_.loc = bc.loc # for printing purposes only

                    bc.insert_before(bc_)

                    # Move jumps over to the PhiNode
                    if bc in self.incoming_jumps:
                        self.incoming_jumps[bc_] = self.incoming_jumps[bc]
                        for bc__ in self.incoming_jumps[bc_]:
                            bc__.setTargetBytecode(bc_)
                        del self.incoming_jumps[bc]

                    logging.debug("IF ADD  " + bc_.locStr())
                    #import pdb; pdb.set_trace()

    def stack_to_register(self):
        logging.debug("Stack->Register Conversion ------------------------------")
        stack = Stack()
        self.todo.push((self.bytecodes, stack))
        evaled = set()

        # For the STORE_FAST of the argument(s)
        for arg in reversed(self.impl.arg_names):
            stack.push(self.impl.getRegister('__param_'+arg))

        while not self.todo.empty():
            (bc, stack) = self.todo.pop()

            if isinstance(bc, Block):
                bc = bc.blockContent()

            r = bc.stack_eval(self.impl, stack)
            evaled.add(bc)
            if r == None:
                # default case: no control flow diversion, just continue with the next
                # instruction in the list
                # Note: the `and not' part is a basic form of dead code elimination
                #       This is used to drop unreachable "return None" which are implicitly added
                #       by Python to the end of functions.
                #       TODO is this the proper way to handle those returns? Any side effects?
                #            NEEDS REVIEW
                #       See also codegen.Program.__init__
                if bc.linearNext() and not isinstance(bc, BlockTerminal):
                    self.todo.push((bc.linearNext(), stack))
                    if isinstance(bc, Block):
                        # the next instruction after the block is now already on the todo list,
                        # but first lets work inside the block
                        self.todo.push((bc.blockContent(), stack))
                else:
                    logging.debug("Reached EOP.")
                    assert stack.empty()
            else:
                # there is (one or more) control flow changes, add them all to the todo list
                assert isinstance(r, list)
                for (bc_, stack_) in r:
                    # don't go back to a bytecode that we already evaluated
                    # if there are no changes on the stack
                    if stack_.empty() and bc_ in evaled:
                        continue
                    self.todo.push((bc_, stack_))

    def type_analysis(self):
        logging.debug("Type Analysis ------------------------------")
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
                    self.retype(self.impl.result.unify_type(bc.result.type, bc.debuginfo))
                if isinstance(bc, BlockTerminal) and bc.linearNext() != None and bc.linearNext() not in self.incoming_jumps:
                    logging.debug("Unreachable {0}, aborting".format(bc.linearNext()))
                    break

            if self.analyze_again:
                self.todo.push(bc_list)

            if i > 10:
                raise Exception("Stopping after {0} type analysis iterations (failsafe)".format(i))
            i += 1

        #logging.debug("last bytecode: " + str(self.bytecodes[-1]))
        logging.debug("returning type " + str(self.impl.result.type))


    def remove(self, bc):
        #import pdb; pdb.set_trace()

        # TODO: should any of these .next become .linearNext()?
        if bc == self.bytecodes:
            self.bytecodes = bc.next

        if bc in self.incoming_jumps:
            bc_next = bc.next
            if not bc_next and bc._block_parent:
                bc_next = bc._block_parent.next
                # _block_parent will be move with bc.remove() below
            assert bc_next
            self.incoming_jumps[bc_next] = self.incoming_jumps[bc]
            for bc_ in self.incoming_jumps[bc_next]:
                bc_.updateTargetBytecode(bc, bc_next)
            del self.incoming_jumps[bc]
        bc.remove()

    def analyze(self, *args):
        self.impl.analyze(args)

        logging.debug("Analysis of " + str(self.impl))

        self.disassemble()

        self.rewrite()

        self.intraflow()

        self.bytecodes.printAll()

        self.stack_to_register()

        self.type_analysis()

        #logging.debug("PyStack bytecode:")
        #import pdb; pdb.set_trace()
        #for bc in self.bytecodes:
        #    logging.debug(str(bc))

    def disassemble(self):
        """Disassemble a code object."""
        logging.debug("Disassembling ------------------------------")

        self.last_bc = None

        # Store arguments in memory locations for uniformity
        for arg in self.impl.arg_names:
            # TODO Patch up di?
            di = None
            bc = STORE_FAST(self.impl, di)
            bc.addArgByName(self.impl, arg)
            if self.last_bc == None:
                self.bytecodes = self.last_bc = bc
            else:
                self.last_bc.insert_after(bc)
                self.last_bc = bc
            logging.debug("DIS'D {0}".format(bc.locStr()))

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
        self.blocks = Stack('blocks')
        while i < n:
            op = code[i]
            if i in linestarts:
                line = linestarts[i]

            di = DebugInfo(co.co_filename, line)

            if extended_arg == 0 and op in opconst:
                bc = opconst[op](self.impl, di)
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
                        bc.addName(co.co_names[oparg])
                    elif op in dis.hasjrel:
                        #print('(to ' + repr(i + oparg) + ')', end=' ')
                        bc.setTarget(i+oparg)
                    elif op in dis.hasjabs:
                        #print(repr(oparg).rjust(5), end=' ')
                        bc.setTarget(oparg)
                    elif op in dis.haslocal:
                        #print('(' + co.co_varnames[oparg] + ')', end=' ')
                        bc.addArgByName(self.impl, co.co_varnames[oparg])
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

            if isinstance(bc, BlockStart):
                # Start of a block.
                # The current bc gets added as the first within the block
                block = Block(bc)
                self.blocks.push(block)
                # Then handle the block as any regular bytecode
                # so that it will be registered appropriately
                bc = block
                # Note the instance(bc, Block) below

            if self.last_bc == None:
                self.bytecodes = bc
            else:
                self.last_bc.insert_after(bc)
            self.last_bc = bc

            if isinstance(bc, Block):
                # Block is inserted, now switch back to appending to the block content
                self.last_bc = bc.blockContent()
            elif isinstance(bc, BlockEnd):
                # Block end, install the block itself as last_bc
                # so that the next instruction is added outside tho block
                self.last_bc = self.blocks.pop()
                # mark the instruction as being the last of the block
                bc.blockEnd(self.last_bc)

def main(f, *args):
    module = bytecode.Module()
    f = Function(f, module)
    f.analyze(*args)
    return f

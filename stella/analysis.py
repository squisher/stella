import dis
import weakref

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
    funcs = weakref.WeakValueDictionary()
    @classmethod
    def get(klass, impl, module):
        logging.debug("Function.get({0}|{1}, {2}|{3})".format(impl, id(impl), module, id(module)))
        try:
            return klass.funcs[(impl, module)]
        except KeyError:
            self = klass(impl, module)
            klass.funcs[(impl, module)] = self
            return self

    def __init__(self, impl, module):
        self.bytecodes = None # pointer to the first bytecode
        self.labels = {}
        self.incoming_jumps = {}

        self.f = impl.f
        self.impl = impl
        self.module = module

        self.log = logging.getLogger(str(self))
        self.todo = Stack("Todo", log=self.log)
        logging.info("Analyzing {0}".format(self))

    def getName(self):
        return str(self.impl)
    def __str__(self):
        return self.getName()

    def retype(self, go = True):
        if go:
            self.analyze_again = True

    def add_incoming_jump(self, target_bc, source_bc):
        if target_bc in self.incoming_jumps:
            self.incoming_jumps[target_bc].append(source_bc)
        else:
            self.incoming_jumps[target_bc] = [source_bc]

    def addLabel(self, bc):
        """Remove replaceLocation() below?"""
        self.replaceLocation(bc)

    def replaceLocation(self, bc):
        """Assume that bc.loc points to the new location already."""
        self.labels[bc.loc] = bc

    def rewrite(self):
        self.bytecodes.printAll(self.log)
        self.log.debug("Rewriting (peephole optimizations) ------------------------------")
        for bc in self.bytecodes:
            try:
                if isinstance(bc, FOR_ITER):
                    # TODO: move this into bytecode.ForLoop
                    cur = bc.prev
                    if not isinstance(cur, GET_ITER):
                        raise UnimplementedError('unsupported for loop')
                    cur.remove()
                    cur = bc.prev
                    if not isinstance(cur, CALL_FUNCTION):
                        raise UnimplementedError('unsupported for loop')
                    cur.remove()
                    cur = bc.prev
                    # TODO: this if..elif should be more general!
                    if isinstance(cur, LOAD_FAST):
                        limit = cur.args[0]
                    elif isinstance(cur, LOAD_CONST):
                        limit = cur.args[0]
                    else:
                        raise UnimplementedError('unsupported for loop')
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
            except StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

        self.bytecodes.printAll(self.log)


    def intraflow(self):
        self.log.debug("Building Intra-Flowgraph ------------------------------")
        for bc in self.bytecodes:
            try:
                if isinstance(bc, Jump):
                    if bc.processFallThrough():
                        self.add_incoming_jump(bc.linearNext(), bc)
                    target_bc = self.labels[bc.target_label]
                    bc.setTargetBytecode(target_bc)
                    self.add_incoming_jump(target_bc, bc)
            except StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

        for bc in self.bytecodes:
            try:
                if bc in self.incoming_jumps:
                    if not isinstance(bc.linearPrev(), BlockTerminal):
                        #import pdb; pdb.set_trace()
                        #self.log.debug("PREV_TYPE " + str(type(bc.prev)))
                        bc_ = Jump(self, bc.debuginfo)
                        bc_.loc = ''
                        bc_.setTargetBytecode(bc)
                        bc_.setTarget(bc.loc) # for printing purposes only
                        bc.insert_before(bc_)
                        self.add_incoming_jump(bc, bc_)

                        self.log.debug("IF ADD  " + bc_.locStr())

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

                        self.log.debug("IF ADD  " + bc_.locStr())
                        #import pdb; pdb.set_trace()
            except StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

    def stack_to_register(self):
        self.log.debug("Stack->Register Conversion ------------------------------")
        stack = Stack(log=self.log)
        self.todo.push((self.bytecodes, stack))
        evaled = set()

        # For the STORE_FAST of the argument(s)
        for arg in reversed(self.impl.arg_transfer):
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
                    # the PhiNode swallows different control flow paths, therefore do not evaluate beyond more than once
                    if not (isinstance(bc, PhiNode) and bc.linearNext() in evaled):
                        self.todo.push((bc.linearNext(), stack))

                    if isinstance(bc, Block):
                        # the next instruction after the block is now already on the todo list,
                        # but first lets work inside the block
                        self.todo.push((bc.blockContent(), stack))
                else:
                    self.log.debug("Reached EOP.")
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
        self.log.debug("Type Analysis ------------------------------")
        self.todo.push(self.bytecodes)

        i = 0
        while not self.todo.empty():
            self.log.debug("Type analysis iteration {0}".format(i))
            self.analyze_again = False
            bc_list = self.todo.pop()

            for bc in bc_list:
                try:
                    bc.type_eval(self)
                    self.log.debug("TYPE'D " + str(bc))
                    if isinstance(bc, RETURN_VALUE):
                        self.retype(self.impl.result.unify_type(bc.result.type, bc.debuginfo))
                    if isinstance(bc, BlockTerminal) and bc.linearNext() != None and bc.linearNext() not in self.incoming_jumps:
                        self.log.debug("Unreachable {0}, aborting".format(bc.linearNext()))
                        break
                except StellaException as e:
                    e.addDebug(bc.debuginfo)
                    raise

            if self.analyze_again:
                self.todo.push(bc_list)

            if i > 10:
                raise Exception("Stopping after {0} type analysis iterations (failsafe)".format(i))
            i += 1

        #self.log.debug("last bytecode: " + str(self.bytecodes[-1]))
        self.log.debug("returning type " + str(self.impl.result.type))


    def analyzeCall(self, args, kwargs):
        self.log.debug("analysis.Function id " + str(id(self)))
        if not self.impl.analyzed:
            self.impl.setParamTypes(args, kwargs)

            self.log.debug("Analysis of " + self.impl.nameAndType())

            self.disassemble()

            self.rewrite()

            self.intraflow()

            self.bytecodes.printAll(self.log)

            self.stack_to_register()

            self.type_analysis()

            self.impl.bytecodes = self.bytecodes
            self.impl.incoming_jumps = self.incoming_jumps

            #self.log.debug("PyStack bytecode:")
            #import pdb; pdb.set_trace()
            #for bc in self.bytecodes:
            #    self.log.debug(str(bc))
        else:
            self.log.debug("Re-typing " + self.impl.nameAndType())

            self.type_analysis()

    def disassemble(self):
        """Disassemble a code object."""
        self.log.debug("Disassembling ------------------------------")

        self.last_bc = None

        # Store arguments in memory locations for uniformity
        for arg in self.impl.arg_transfer:
            # TODO Patch up di?
            di = None
            bc = STORE_FAST(self.impl, di)
            bc.addLocalName(self.impl, arg)
            if self.last_bc == None:
                self.bytecodes = self.last_bc = bc
            else:
                self.last_bc.insert_after(bc)
                self.last_bc = bc
            self.log.debug("DIS'D {0}".format(bc.locStr()))

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
                #if isinstance(bc, CALL_FUNCTION):
                #    pdb.set_trace()
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
                        bc.addName(self.impl, co.co_names[oparg])
                    elif op in dis.hasjrel:
                        #print('(to ' + repr(i + oparg) + ')', end=' ')
                        bc.setTarget(i+oparg)
                    elif op in dis.hasjabs:
                        #print(repr(oparg).rjust(5), end=' ')
                        bc.setTarget(oparg)
                    elif op in dis.haslocal:
                        #print('(' + co.co_varnames[oparg] + ')', end=' ')
                        bc.addLocalName(self.impl, co.co_varnames[oparg])
                    elif op in dis.hascompare:
                        #print('(' + dis.cmp_op[oparg] + ')', end=' ')
                        bc.addCmp(dis.cmp_op[oparg])
                    elif op in dis.hasfree:
                        if free is None:
                            free = co.co_cellvars + co.co_freevars
                        #print('(' + free[oparg] + ')', end=' ')
                        raise UnimplementedError('hasfree')
                    else:
                        bc.addRawArg(oparg)

                self.log.debug("DIS'D {0}".format(bc.locStr()))
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

def main(f, args, kwargs):
    module = bytecode.Module()
    impl = bytecode.Function(f, module)

    const_kw = {}
    for k,v in kwargs.items():
        const_kw[k] = Const(v)
    impl.makeEntry(list(map(wrapValue, args)), const_kw)
    module.addFunc(impl)

    f = Function.get(impl, module)
    f.analyzeCall(args, kwargs)
    f.log.debug("called functions: " + str(module.todoCount()))
    while module.todoCount() > 0:
        # TODO add kwargs support!
        #pdb.set_trace()
        (call_impl, call_args, call_kwargs) = module.todoNext()
        call_f = Function.get(call_impl, module)
        call_f.analyzeCall(call_args, call_kwargs)
    return module

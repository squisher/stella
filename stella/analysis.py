# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dis
import logging
import inspect

from . import exc
from . import bytecode
from . import ir
from . import tp
from . import utils


class DebugInfo(object):
    line = None
    filename = None

    def __init__(self, filename, line):
        self.line = line
        self.filename = filename

    def __str__(self):
        return self.filename + ':' + str(self.line)


class Function(object):
    funcs = {}
    analysis_count = 0

    @classmethod
    def clearCache(klass):
        klass.funcs.clear()

    @classmethod
    def get(klass, f, module):
        if isinstance(f, ir.FunctionRef):
            impl = f.function
        elif isinstance(f, ir.Function):
            impl = f
        else:
            raise exc.TypeError("{} is not a Function, it has type {}".format(f, type(f)))

        logging.debug("Function.get({0}|{1}, {2})".format(
            impl, id(impl), module))
        try:
            return klass.funcs[(impl, module)]
        except KeyError:
            self = klass(impl, module)
            klass.funcs[(impl, module)] = self
            return self

    def __init__(self, impl, module):
        self.bytecodes = None  # pointer to the first bytecode
        self.labels = {}
        self.incoming_jumps = {}

        self.f = impl.pyFunc()
        self.impl = impl
        self.module = module

        self.log = logging.getLogger(str(self))
        self.todo = utils.Stack("Todo", log=self.log, quiet=True)
        logging.info("Analyzing {0}".format(self))

    def __str__(self):
        return str(self.impl)

    def __repr__(self):
        return "{}:{}>".format(super().__repr__()[:-1], self)

    def retype(self, go=True):
        """Immediately retype this function if go is True"""
        if isinstance(go, tuple):
            # extract the widening
            go = go[0]
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
        self.log.debug("Rewriting (peephole optimizations) " + '-'*40)
        for bc in self.bytecodes:
            try:
                if isinstance(bc, bytecode.FOR_ITER):
                    for_loop = bytecode.ForLoop(self, bc.debuginfo)
                    for_loop.basicSetup(bc)
                    for_loop.rewrite(self)
                    self.replaceLocation(for_loop)
            except exc.StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

    def intraflow(self):
        self.log.debug("Building Intra-Flowgraph " + '-'*40)
        for bc in self.bytecodes:
            try:
                if isinstance(bc, bytecode.Jump):
                    if bc.processFallThrough():
                        self.add_incoming_jump(bc.linearNext(), bc)
                if isinstance(bc, bytecode.HasTarget):
                    target_bc = self.labels[bc.target_label]
                    bc.setTargetBytecode(target_bc)
                    self.add_incoming_jump(target_bc, bc)
            except exc.StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

        for bc in self.bytecodes:
            try:
                if bc in self.incoming_jumps:
                    bc_prev = bc.linearPrev()
                    # TODO Ugly -- blocks aren't transparent enough
                    if isinstance(bc_prev, utils.Block):
                        bc_prev = bc_prev.blockContent()
                    if bc_prev and not isinstance(bc_prev, utils.BlockTerminal):
                        bc_ = bytecode.Jump(self, bc.debuginfo)
                        bc_.loc = ''
                        bc_.setTargetBytecode(bc)
                        bc_.setTarget(bc.loc)  # for printing purposes only
                        bc.insert_before(bc_)
                        self.add_incoming_jump(bc, bc_)

                        self.log.debug("IF ADD  " + bc_.locStr())

                    if len(self.incoming_jumps[bc]) > 1:
                        bc_ = ir.PhiNode(self.impl, bc.debuginfo)
                        bc_.loc = bc.loc  # for printing purposes only

                        bc.insert_before(bc_)

                        # Move jumps over to the PhiNode
                        if bc in self.incoming_jumps:
                            self.incoming_jumps[bc_] = self.incoming_jumps[bc]
                            for bc__ in self.incoming_jumps[bc_]:
                                bc__.setTargetBytecode(bc_)
                            del self.incoming_jumps[bc]

                        self.log.debug("IF ADD  " + bc_.locStr())
            except exc.StellaException as e:
                e.addDebug(bc.debuginfo)
                raise

    def stack_to_register(self):
        self.log.debug("Stack->Register Conversion " + '-'*40)
        stack = utils.Stack(log=self.log)
        self.todo.push((self.bytecodes, stack))
        evaled = set()

        # For the STORE_FAST of the argument(s)
        for arg in reversed(self.impl.arg_transfer):
            arg_bc = bytecode.ResultOnlyBytecode(self.impl, None)
            arg_bc.result = self.impl.getRegister('__param_' + arg)
            stack.push(arg_bc)

        while not self.todo.empty():
            (bc, stack) = self.todo.pop()

            if isinstance(bc, utils.Block):
                bc = bc.blockContent()

            r = bc.stack_eval(self.impl, stack)
            evaled.add(bc)
            if r is None:
                # default case: no control flow diversion, just continue with
                # the next instruction in the list
                # Note: the `and not' part is a basic form of dead code
                # elimination. This is used to drop unreachable "return None"
                # which are implicitly added by Python to the end of functions.
                # TODO is this the proper way to handle those returns? Any side
                # effects?
                # NEEDS REVIEW See also codegen.Program.__init__
                if bc.linearNext() and not isinstance(bc, utils.BlockTerminal):
                    # the PhiNode swallows different control flow paths,
                    # therefore do not evaluate beyond more than once
                    if not (isinstance(bc, ir.PhiNode) and
                            bc.linearNext() in evaled):
                        self.todo.push((bc.linearNext(), stack))

                    if isinstance(bc, utils.Block):
                        # the next instruction after the block is now already
                        # on the todo list, but first lets work inside the block
                        self.todo.push((bc.blockContent(), stack))
                else:
                    if bc.linearNext() and not self.todo.contains(lambda x: x[0] == bc.linearNext()):
                        # continue processing, because python does generate
                        # dead code, unless that code is already on our TODO
                        # list
                        self.log.debug("Next instruction is not directly reachable, but continuing anyway")
                        self.todo.push((bc.linearNext(), stack))
                    else:
                        assert stack.empty()
            else:
                # there is (one or more) control flow changes, add them all to
                # the todo list
                assert isinstance(r, list)
                for (bc_, stack_) in r:
                    # don't go back to a bytecode that we already evaluated
                    # if there are no changes on the stack
                    if stack_.empty() and bc_ in evaled:
                        continue
                    self.todo.push((bc_, stack_))

    def type_analysis(self):
        self.log.debug("Type Analysis " + '-'*40)
        self.todo.push(self.bytecodes)

        i = 0
        reachable = True
        while not self.todo.empty():
            self.log.debug("Type analysis iteration {0}".format(i))
            self.analyze_again = False
            bc_list = self.todo.pop()

            for bc in bc_list:
                try:
                    if reachable:
                        abort = bc.type_eval(self)
                        self.log.debug("TYPE'D " + bc.locStr())
                        if isinstance(bc, bytecode.RETURN_VALUE):
                            self.retype(self.impl.result.unify_type(bc.result.type,
                                                                bc.debuginfo))
                    else:
                        self.log.debug("UNTYPE " + bc.locStr() + " -- unreachable")
                        bc.reachable = False

                    if isinstance(bc, utils.BlockTerminal) and \
                            bc.linearNext() is not None and \
                            bc.linearNext() not in self.incoming_jumps:
                        # Python does generate unreachable code which is then
                        # followed by reachable code. This allows us to jump
                        # over the section that needs to be ignored
                        self.log.debug("Unreachable {0}, but continuing".format(bc.linearNext()))
                        reachable = False
                    elif bc.linearNext() in self.incoming_jumps:
                        # Once there is an incoming jump, the following code is
                        # reachable again
                        reachable = True

                    if abort:
                        self.log.debug("Aborting typing, resuming later.")
                        self.log.debug("{!r}".format(self.todo))
                        self.impl.analyzeAgain()
                        break
                except exc.StellaException as e:
                    e.addDebug(bc.debuginfo)
                    raise

            if self.analyze_again:
                self.todo.push(bc_list)

            if i > 3:
                raise Exception("Stopping after {0} type analysis iterations (failsafe)".format(i))
            i += 1

        self.log.debug("returning type " + str(self.impl.result.type))

    def analyzeCall(self, args, kwargs):
        self.log.debug("analysis.Function id " + str(id(self)))
        if not self.impl.analyzed:
            self.impl.setupArgs(args, kwargs)

            self.log.debug("Analysis of " + self.impl.nameAndType())

            self.disassemble()
            self.bytecodes.printAll(self.log)

            self.rewrite()
            self.bytecodes.printAll(self.log)

            self.intraflow()
            self.bytecodes.printAll(self.log)

            self.stack_to_register()

            self.type_analysis()

            self.impl.bytecodes = self.bytecodes
            self.impl.incoming_jumps = self.incoming_jumps
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
            bc = bytecode.STORE_FAST(self.impl, di)
            bc.addLocalName(self.impl, arg)
            if self.last_bc is None:
                self.bytecodes = self.last_bc = bc
            else:
                self.last_bc.insert_after(bc)
                self.last_bc = bc
            self.log.debug("DIS'D {0}".format(bc.locStr()))

        co = self.f.__code__
        code = co.co_code
        labels = dis.findlabels(code)
        linestarts = dict(dis.findlinestarts(co))
        n = len(code)
        i = 0
        extended_arg = 0
        free = None
        line = 0
        self.blocks = utils.Stack('blocks')
        while i < n:
            op = code[i]
            if i in linestarts:
                line = linestarts[i]

            di = DebugInfo(co.co_filename, line)

            if extended_arg == 0 and op in bytecode.opconst:
                bc = bytecode.opconst[op](self.impl, di)
            else:
                raise exc.UnsupportedOpcode(op, di)
            bc.loc = i

            if i in labels:
                self.labels[i] = bc

            # print(repr(i).rjust(4), end=' ')
            # print(dis.opname[op].ljust(20), end=' ')
            i = i + 1
            try:
                if op >= dis.HAVE_ARGUMENT:
                    oparg = code[i] + code[i + 1]*256 + extended_arg
                    extended_arg = 0
                    i = i+2
                    if op == dis.EXTENDED_ARG:
                        extended_arg = oparg*65536

                    if op in dis.hasconst:
                        # print('(' + repr(co.co_consts[oparg]) + ')', end=' ')
                        bc.addConst(co.co_consts[oparg])
                    elif op in dis.hasname:
                        # print('(' + co.co_names[oparg] + ')', end=' ')
                        bc.addName(self.impl, co.co_names[oparg])
                    elif op in dis.hasjrel:
                        # print('(to ' + repr(i + oparg) + ')', end=' ')
                        bc.setTarget(i + oparg)
                    elif op in dis.hasjabs:
                        # print(repr(oparg).rjust(5), end=' ')
                        bc.setTarget(oparg)
                    elif op in dis.haslocal:
                        # print('(' + co.co_varnames[oparg] + ')', end=' ')
                        bc.addLocalName(self.impl, co.co_varnames[oparg])
                    elif op in dis.hascompare:
                        # print('(' + dis.cmp_op[oparg] + ')', end=' ')
                        bc.addCmp(dis.cmp_op[oparg])
                    elif op in dis.hasfree:
                        if free is None:
                            free = co.co_cellvars + co.co_freevars
                        # print('(' + free[oparg] + ')', end=' ')
                        raise exc.UnimplementedError('hasfree')
                    else:
                        bc.addRawArg(oparg)

                self.log.debug("DIS'D {0}".format(bc.locStr()))
            except exc.StellaException as e:
                e.addDebug(di)
                raise

            if isinstance(bc, utils.BlockStart):
                # Start of a block.
                # The current bc gets added as the first within the block
                block = utils.Block(bc)
                self.blocks.push(block)
                # Then handle the block as any regular bytecode
                # so that it will be registered appropriately
                bc = block
                # Note the instance(bc, Block) below

            if self.last_bc is None:
                self.bytecodes = bc
            else:
                self.last_bc.insert_after(bc)
            self.last_bc = bc

            if isinstance(bc, utils.Block):
                # Block is inserted, now switch back to appending to the block
                # content
                self.last_bc = bc.blockContent()
            elif isinstance(bc, utils.BlockEnd):
                # Block end, install the block itself as last_bc
                # so that the next instruction is added outside tho block
                self.last_bc = self.blocks.pop()
                # mark the instruction as being the last of the block
                bc.blockEnd(self.last_bc)


def cleanup():
    logging.debug("Cleaning up...")
    Function.clearCache()
    tp.destruct()


def main(f, args, kwargs):
    # Clean up first since an internal failure may have prevented the
    # destructors from running.
    cleanup()

    try:
        module = ir.Module()
        f_type = tp.get(f)
        funcref = module.getFunctionRef(f_type)

        if f_type.bound:
            f_self = tp.wrapValue(f.__self__)
            funcref.f_self = f_self

        # TODO: why do I use wrapValue for args but Const for kwargs...?
        const_kw = {}
        for k, v in kwargs.items():
            const_kw[k] = tp.Const(v)
        funcref.makeEntry(list(map(tp.wrapValue, args)), const_kw)

        f = Function.get(funcref, module)

        wrapped_args = [tp.wrapValue(arg) for arg in args]
        wrapped_kwargs = {k: tp.wrapValue(v) for k, v in kwargs.items()}

    except exc.StellaException as e:
        # An error occurred while preparing the entry call, so at this point
        # it's best to attribute it to the caller
        (frame, filename, line_number,
         function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[2]
        debuginfo = DebugInfo(filename, line_number)
        e.addDebug(debuginfo)
        raise e

    f.analyzeCall(wrapped_args, wrapped_kwargs)

    while module.todoCount() > 0:
        f.log.debug("called functions: {} ({})".format(module.todoList(), module.todoCount()))
        # TODO add kwargs support!
        (call_impl, call_args, call_kwargs) = module.todoNext()
        call_f = Function.get(call_impl, module)
        if call_f.analysis_count > 30:
            # TODO: abitrary limit, it would be better to check if the return
            # type changed or not
            raise Exception("Stopping {} after {} call analysis iterations (failsafe)".format(
                call_f, call_f.analysis_count))
        call_f.analyzeCall(call_args, call_kwargs)
        call_f.analysis_count += 1
    module.addDestruct(cleanup)
    return module

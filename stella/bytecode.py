import dis
import logging
import sys
import types

import llvm
import llvm.core
import llvm.ee

from . import tp
from .exc import *
from .utils import *
from .intrinsics import *
from .ir import *

def pop_stack(n):
    """
    Decorator, it takes n items off the stack
    and adds them as bytecode arguments.
    """
    def extract_n(f):
        def extract_from_stack(self, func, stack):
            args = []
            for i in range(n):
                args.append(stack.pop())
            args.reverse()

            if self.args == None:
                self.args = args
            else:
                self.args.extend(args)
            return f(self, func, stack)
        return extract_from_stack
    return extract_n

class BlockTerminal(object):
    """
    Marker class for instructions which terminate a block.
    """
    pass

class Poison(object):
    """
    Require that this bytecode is rewritten by bailing out
    if it is ever evaluated.

    Note that if the child overrides all methods, this mixin will be useless
    and should be removed from the child.
    """
    def stack_eval(self, func, stack):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))

    def translate(self, module, builder):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))

    def type_eval(self, func):
        raise UnimplementedError("{0} must be rewritten".format(self.__class__.__name__))


class Bytecode(IR):
    """
    Parent class for all Python bytecodes
    """
    pass

class LOAD_FAST(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)


    def addLocalName(self, func, name):
        # TODO: crude?
        try:
            self.args.append(func.getRegister(name))
        except UndefinedError:
            self.args.append(func.getStackLoc(name))

    def stack_eval(self, func, stack):
        type_ = type(self.args[0])
        if type_ == StackLoc:
            self.result = Register(func)
        elif type_ == Register:
            self.result = self.args[0]
        else:
            raise StellaException("Invalid LOAD_FAST argument type `{0}'".format(type_))
        stack.push(self.result)

    def type_eval(self, func):
        self.result.type = self.args[0].type

    def translate(self, module, builder):
        type_ = type(self.args[0])
        if type_ == StackLoc:
            self.result.llvm = builder.load(self.args[0].llvm)
        elif type_ == Register:
            # nothing to load, it's a pseudo instruction in this case
            pass

class STORE_FAST(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.new_allocate = False

    def addLocalName(self, func, name):
        # Python does not allocate new names, it just refers to them
        #import pdb; pdb.set_trace()
        (var, self.new_allocate) = func.getOrNewStackLoc(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        self.result = self.popFirstArg()

    def type_eval(self, func):
        #func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        arg = self.args[0]
        #import pdb; pdb.set_trace()
        tp_changed = self.result.unify_type(arg.type, self.debuginfo)
        if tp_changed:
            # TODO: can I avoid a retype in some cases?
            func.retype()
            if self.result.type != arg.type:
                self.args[0] = Cast(arg, self.result.type)

    def translate(self, module, builder):
        self.cast(builder)
        if self.new_allocate:
            type_ = self.args[0].llvmType()
            # TODO: is this the right place to make it a pointer?
            if type(self.args[0].type) == tp.ArrayType:
                type_ = llvm.core.Type.pointer(type_)
            self.result.llvm = builder.alloca(type_, name=self.result.name)
        builder.store(self.args[0].llvm, self.result.llvm)

class STORE_GLOBAL(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        # Python does not allocate new names, it just refers to them
        #import pdb; pdb.set_trace()
        var = func.loadGlobal(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        self.result = self.popFirstArg()

    def type_eval(self, func):
        #func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        arg = self.args[0]
        #import pdb; pdb.set_trace()
        tp_changed = self.result.unify_type(arg.type, self.debuginfo)
        if tp_changed:
            # TODO: can I avoid a retype in some cases?
            func.retype()
            if self.result.type != arg.type:
                self.args[0] = Cast(arg, self.result.type)

    def translate(self, module, builder):
        # Assume that the global has been allocated already.
        self.cast(builder)
        builder.store(self.args[0].llvm, self.result.llvm)

class LOAD_CONST(Bytecode):
    discard = True
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        if self.result == None:
            self.result = self.popFirstArg()
        stack.push(self.result)

    def type_eval(self, func):
        pass
    def translate(self, module, builder):
        pass

class BinaryOp(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self.result)

    def type_eval(self, func):
        for i in range(len(self.args)):
            arg = self.args[i]
            tp_changed = self.result.unify_type(arg.type, self.debuginfo)
            if tp_changed:
                if self.result.type != arg.type:
                    self.args[i] = Cast(arg, self.result.type)
                # TODO: can I avoid a retype in some cases?
                func.retype()

    def builderFuncName(self):
        try:
            return self.b_func[self.result.type]
        except KeyError:
            raise TypingError("{0} does not yet implement type {1}".format(self.__class__.__name__, self.result.type))

    def translate(self, module, builder):
        self.cast(builder)
        f = getattr(builder, self.builderFuncName())
        self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

    @abstractproperty
    def b_func(self):
        return {}

class BINARY_ADD(BinaryOp):
    b_func = {tp.Float: 'fadd', tp.Int: 'add'}

class BINARY_SUBTRACT(BinaryOp):
    b_func = {tp.Float: 'fsub', tp.Int: 'sub'}

class BINARY_MULTIPLY(BinaryOp):
    b_func = {tp.Float: 'fmul', tp.Int: 'mul'}

class BINARY_MODULO(BinaryOp):
    b_func = {tp.Float: 'frem', tp.Int: 'srem'}

class BINARY_POWER(BinaryOp):
    b_func = {tp.Float: llvm.core.INTR_POW, tp.Int: llvm.core.INTR_POWI}

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        # TODO if args[1] is int but negative, then the result will be float, too!
#        if self.args[0].type == tp.Int and self.args[1].type == tp.Int:
#            type_ = tp.Int
#        else:
#            type_ = tp.Float
#        tp_changed = self.result.unify_type(type_, self.debuginfo)
        super().type_eval(func)

    def translate(self, module, builder):
        # llvm.pow[i]'s first argument always has to be float
        if self.args[0].type == tp.Int:
            self.args[0] = Cast(self.args[0], tp.Float)

        self.cast(builder)

        if self.args[1].type == tp.Int:
            # powi takes a i32 argument
            power = builder.trunc(self.args[1].llvm, tp.tp_int32, '(i32)'+self.args[1].name)
        else:
            power = self.args[1].llvm

        llvm_pow = llvm.core.Function.intrinsic(module, self.b_func[self.args[1].type], [self.args[0].llvmType()])
        pow_result = builder.call(llvm_pow, [self.args[0].llvm, power])

        if isinstance(self.args[0], Cast) and self.args[0].obj.type == tp.Int and self.args[1].type == tp.Int:
            # cast back to an integer
            self.result.llvm = builder.fptosi(pow_result, tp.Int.llvmType())
        else:
            self.result.llvm = pow_result

class BINARY_FLOOR_DIVIDE(BinaryOp):
    """Python compliant `//' operator: Slow since it has to perform type conversions and floating point division for integers"""
    b_func = {tp.Float: 'fdiv', tp.Int: 'fdiv'} # NOT USED, but required to make it a concrete class

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def type_eval(self, func):
        for arg in self.args:
            # TODO: this is a HACK
            if isinstance(arg, Cast):
                type_ = arg.obj.type
            else:
                type_ = arg.type
            self.result.unify_type(type_, self.debuginfo)

        # convert all arguments to float, since fp division is required to apply floor
        for i in range(len(self.args)):
            arg = self.args[i]
            do_cast = arg.type != tp.Float
            if do_cast:
                self.args[i] = Cast(arg, tp.Float)
                func.retype()

    def translate(self, module, builder):
        self.cast(builder)

        tmp = builder.fdiv(self.args[0].llvm, self.args[1].llvm, self.result.name)
        llvm_floor = llvm.core.Function.intrinsic(module, llvm.core.INTR_FLOOR, [tp.Float.llvmType()])
        self.result.llvm = builder.call(llvm_floor, [tmp])

        #import pdb; pdb.set_trace()
        if all([isinstance(a,Cast) and a.obj.type == tp.Int for a in self.args]):
            # TODO this may be superflous if both args got converted to float in the translation stage -> move toFloat partially to the analysis stage.
            self.result.llvm = builder.fptosi(self.result.llvm, tp.Int.llvmType(), "(int)"+self.result.name)

class BINARY_TRUE_DIVIDE(BinaryOp):
    b_func = {tp.Float: 'fdiv'}

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        # The result of `/', true division, is always a float
        self.result.type = tp.Float
        super().type_eval(func)

#class InplaceOp(BinaryOp):
#    @pop_stack(2)
#    def stack_eval(self, func, stack):
#        type_ = unify_type(self.args[0].type, self.args[1].type, self.debuginfo)
#        if type_ != self.args[0].type:
#            self.result = Local.tmp(self.args[0])
#            self.result.type = tp
#        else:
#            self.result = self.args[0]
#        stack.push(self.result)
#
#    def translate(self, module, builder):
#        self.floatArg(builder)
#        f = getattr(builder, self.builderFuncName())
#        self.result.llvm = f(self.args[0].llvm, self.args[1].llvm, self.result.name)

# Inplace operators don't have a semantic difference when used on primitive types
class INPLACE_ADD(BINARY_ADD): pass
class INPLACE_SUBTRACT(BINARY_SUBTRACT): pass
class INPLACE_MULTIPLY(BINARY_MULTIPLY): pass
class INPLACE_TRUE_DIVIDE(BINARY_TRUE_DIVIDE): pass
class INPLACE_FLOOR_DIVIDE(BINARY_FLOOR_DIVIDE): pass
class INPLACE_MODULO(BINARY_MODULO): pass

class COMPARE_OP(Bytecode):
    b_func = {tp.Float: 'fcmp', tp.Int: 'icmp', tp.Bool: 'icmp'}
    op = None

    icmp = {'==': llvm.core.ICMP_EQ,
            '!=': llvm.core.ICMP_NE,
            '>':  llvm.core.ICMP_SGT,
            '>=': llvm.core.ICMP_SGE,
            '<':  llvm.core.ICMP_SLT,
            '<=': llvm.core.ICMP_SLE,
            }

    fcmp = {'==': llvm.core.FCMP_OEQ,
            '!=': llvm.core.FCMP_ONE,
            '>':  llvm.core.FCMP_OGT,
            '>=': llvm.core.FCMP_OGE,
            '<':  llvm.core.FCMP_OLT,
            '<=': llvm.core.FCMP_OLE,
            }

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addCmp(self, op):
        self.op = op

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        #func.retype(self.result.unify_type(bool, self.debuginfo))
        self.result.type = tp.Bool
        if self.args[0].type != self.args[1].type:
            raise TypingError("Comparing different types ({0} with {1})".format(self.args[0].type, self.args[1].type))

    def translate(self, module, builder):
        # assume both types are the same, see @stack_eval
        type_ = self.args[0].type
        if not self.args[0].type in self.b_func:
            raise UnimplementedError(type_)

        f = getattr(builder, self.b_func[type_])
        m = getattr(self,    self.b_func[type_])

        self.result.llvm = f(m[self.op], self.args[0].llvm, self.args[1].llvm, self.result.name)

class RETURN_VALUE(BlockTerminal, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        #import pdb; pdb.set_trace()
        self.result = self.popFirstArg()

    def type_eval(self, func):
        for arg in self.args:
            func.retype(self.result.unify_type(arg.type, self.debuginfo))

    def translate(self, module, builder):
        if self.result.type == tp.Void:
            # return None == void, do not generate a ret instruction as that is invalid
            builder.ret_void()
        else:
            builder.ret(self.result.llvm)

class HasTarget(object):
    target_label = None
    target_bc = None
    def setTargetBytecode(self, bc):
        #import pdb; pdb.set_trace()
        self.target_bc = bc

    def updateTargetBytecode(self, old_bc, new_bc):
        self.setTargetBytecode(new_bc)

    def setTarget(self, label):
        self.target_label = label

    def __str__(self):
        return "{0} {1} {2}".format(
                    self.__class__.__name__,
                    self.target_label,
                    ", ".join([str(v) for v in self.args]))


class Jump(BlockTerminal, HasTarget, IR):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        return False

    def stack_eval(self, func, stack):
        tos = stack.peek()
        if tos:
            tos.bc = self
        return [(self.target_bc, stack)]

    def type_eval(self,func):
        pass

    def translate(self, module, builder):
        builder.branch(self.target_bc.block)

class Jump_if_X_or_pop(Jump):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        self.fallthrough = self.next
        return True

    def updateTargetBytecode(self, old_bc, new_bc):
        if old_bc == self.target_bc:
            self.setTargetBytecode(new_bc)
        else:
            assert self.fallthrough == old_bc
            self.fallthrough = new_bc

    @pop_stack(1)
    def stack_eval(self, func, stack):
        stack2 = stack.clone()
        r = []
        # if X, push back onto stack and jump:
        self.args[0].bc = self
        stack.push(self.args[0])
        r.append((self.target_bc, stack))
        # else continue with the next instruction (and keep the popped value)
        if not stack2.empty():
            # TODO: is stack2 always empty?
            tos = stack2.peek()
            if tos:
                tos.bc = self
        r.append((self.next, stack2))

        return r

class JUMP_IF_FALSE_OR_POP(Jump_if_X_or_pop, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        #import pdb; pdb.set_trace()
        builder.cbranch(self.args[0].llvm, self.next.block, self.target_bc.block)

class JUMP_IF_TRUE_OR_POP(Jump_if_X_or_pop, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.target_bc.block, self.next.block)

class Pop_jump_if_X(Jump):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        self.fallthrough = self.next
        return True

    def updateTargetBytecode(self, old_bc, new_bc):
        if old_bc == self.target_bc:
            self.setTargetBytecode(new_bc)
        else:
            assert self.fallthrough == old_bc
            self.fallthrough = new_bc

    @pop_stack(1)
    def stack_eval(self, func, stack):
        r = []
        # if X, jump
        self.args[0].bc = self
        r.append((self.target_bc, stack))
        # else continue to the next instruction
        r.append((self.next, stack))
        # (pop happens in any case)

        return r

class POP_JUMP_IF_FALSE(Pop_jump_if_X, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.next.block, self.target_bc.block)

class POP_JUMP_IF_TRUE(Pop_jump_if_X, Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, module, builder):
        builder.cbranch(self.args[0].llvm, self.target_bc.block, self.next.block)

class SETUP_LOOP(BlockStart, HasTarget, Bytecode):
    """
    Will either be rewritten (for loop) or has no effect other than mark the
    start of a block (while loop).
    """
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class POP_BLOCK(BlockEnd, Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class LOAD_GLOBAL(Bytecode):
    var = None

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        self.args.append(name)

    def stack_eval(self, func, stack):
        self.var = func.loadGlobal(self.args[0])
        # TODO: remove these isinstance checks and just check for GlobalVariable else return directly?
        if isinstance(self.var, Function):
            self.result = self.var
        elif isinstance(self.var, types.ModuleType):
            self.result = self.var
        elif isinstance(self.var, type):
            self.result = self.var
        elif isinstance(self.var, Intrinsic):
            self.result = self.var
        elif isinstance(self.var, GlobalVariable):
            self.result = Register(func)
        else:
            raise UnimplementedError("Unknown global type {0}".format(type(self.result)))
        stack.push(self.result)

    def translate(self, module, builder):
        if isinstance(self.var, Function):
            pass
        elif isinstance(self.var, GlobalVariable):
            self.result.llvm = builder.load(self.var.llvm)

    def type_eval(self, func):
        if isinstance(self.var, GlobalVariable):
            self.result.unify_type(self.var.type, self.debuginfo)

class LOAD_ATTR(Bytecode):
    discard = True
    var = None

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        self.args.append(name)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        if isinstance(self.args[1], types.ModuleType):
            self.result = func.module.loadExt(self.args[1], self.args[0])
        else:
            raise UnimplementedError("Cannot load attribute {0} of an object with type {1}".format(self.args[0], type(self.args[1])))
        stack.push(self.result)

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        pass

class CALL_FUNCTION(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addRawArg(self, arg):
        self.num_pos_args = arg & 0xFF
        self.num_kw_args = (arg >> 8) & 0xFF

    def separateArgs(self):
        self.func = self.args[0]
        args = self.args[1:]

        #pdb.set_trace()
        assert len(args) == self.num_pos_args + self.num_kw_args*2

        self.kw_args = {}
        for i in range(self.num_kw_args):
            # the list is reversed, so the value comes first
            value = args.pop()
            key = args.pop()
            # key is a Const object, unwrap it
            self.kw_args[key.value] = value

        # remainder is positional
        self.args = args

    def stack_eval(self, func, stack):
        for i in range(self.num_pos_args + 2*self.num_kw_args +1):
            arg = stack.pop()
            self.args.append(arg)
        self.args.reverse()
        self.separateArgs()

        if isinstance(self.func, Intrinsic):
            args = self.func.combineAndCheckArgs(self.args, self.kw_args)
            self.func.addArgs(args)
            self.result = self.func.getResult(func)
        else:
            self.result = Register(func)

            func.module.functionCall(self.func, self.args, self.kw_args)
        stack.push(self.result)

    def type_eval(self, func):
        type_ = self.func.getReturnType()
        tp_change = self.result.unify_type(type_, self.debuginfo)

        if self.result.type == tp.NoType:
            func.impl.analyzeAgain() # redo analysis, right now return type is not known
        else:
            func.retype(tp_change)

    def translate(self, module, builder):
        if isinstance(self.func, Intrinsic):
            self.result.llvm = self.func.translate(module, builder)
        else:
            args = self.func.combineAndCheckArgs(self.args, self.kw_args)
            #logging.debug("Call using args: " + str(args))
            #logging.debug("Call using arg_types: " + str(list(map (type, args))))

            self.result.llvm = builder.call(self.func.llvm, [arg.llvm for arg in args])

class GET_ITER(Poison, Bytecode):
    """WIP"""
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class FOR_ITER(Poison, HasTarget, Bytecode):
    """WIP"""
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class JUMP_ABSOLUTE(Jump, Bytecode):
    """WIP"""
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

class JUMP_FORWARD(Jump, Bytecode):
    """WIP"""
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)


class ForLoop(IR):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def setLoopVar(self, loop_var):
        self.loop_var = loop_var
    def setLimit(self, limit):
        self.limit = limit
    def setEndLoc(self, end_loc):
        self.end_loc = end_loc
    def setTestLoc(self, loc):
        self.test_loc = loc

    def rewrite(self, func):
        last = self
        (self.limit_minus_one,_) = func.impl.getOrNewStackLoc(str(self.test_loc) + "__limit")

        # init
        b = LOAD_CONST(func.impl, self.debuginfo)
        b.addArg(Const(0))
        last.insert_after(b)
        last = b

        b = STORE_FAST(func.impl, self.debuginfo)
        #import pdb; pdb.set_trace()
        b.addArg(self.loop_var)
        b.new_allocate = True
        last.insert_after(b)
        last = b

        # initial test
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        b.loc = self.test_loc
        func.replaceLocation(b)
        last.insert_after(b)
        last = b

        if isinstance(self.limit, StackLoc):
            b = LOAD_FAST(func.impl, self.debuginfo)
        elif isinstance(self.limit, Const):
            b = LOAD_CONST(func.impl, self.debuginfo)
        else:
            raise UnimplementedError("Unsupported limit type {0}".format(type(self.limit)))
        b.addArg(self.limit)
        last.insert_after(b)
        last = b

        b = COMPARE_OP(func.impl, self.debuginfo)
        b.addCmp('>=')
        last.insert_after(b)
        last = b

        b = POP_JUMP_IF_TRUE(func.impl, self.debuginfo)
        b.setTarget(self.end_loc)
        last.insert_after(b)
        last = b

        # my_limit = limit -1
        if isinstance(self.limit, StackLoc):
            b = LOAD_FAST(func.impl, self.debuginfo)
        elif isinstance(self.limit, Const):
            b = LOAD_CONST(func.impl, self.debuginfo)
        else:
            raise UnimplementedError("Unsupported limit type {0}".format(type(self.limit)))
        b.addArg(self.limit)
        last.insert_after(b)
        last = b

        b = LOAD_CONST(func.impl, self.debuginfo)
        b.addArg(Const(1))
        last.insert_after(b)
        last = b

        b = BINARY_SUBTRACT(func.impl, self.debuginfo)
        last.insert_after(b)
        last = b

        b = STORE_FAST(func.impl, self.debuginfo)
        b.addArg(self.limit_minus_one)
        b.new_allocate = True
        last.insert_after(b)
        last = b


        # $body, keep, find the end of it
        body_loc = b.next.loc
        func.addLabel(b.next)

        while b.next != None:
            b = b.next
        assert isinstance(b, BlockEnd)
        #import pdb; pdb.set_trace()
        jump_loc = b.loc
        last = b.prev
        b.remove()

        # go back to the JUMP and switch locations
        incr_loc = last.loc
        last.loc = jump_loc
        func.replaceLocation(last)

        # loop test
        #pdb.set_trace()
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        last.insert_before(b)

        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.limit_minus_one)
        last.insert_before(b)

        b = COMPARE_OP(func.impl, self.debuginfo)
        b.addCmp('>=')
        last.insert_before(b)

        b = POP_JUMP_IF_TRUE(func.impl, self.debuginfo)
        b.setTarget(self.end_loc)
        last.insert_before(b)


        # increment
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        b.loc = incr_loc
        func.replaceLocation(b)
        last.insert_before(b)

        b = LOAD_CONST(func.impl, self.debuginfo)
        b.addArg(Const(1))
        last.insert_before(b)

        b = INPLACE_ADD(func.impl, self.debuginfo)
        last.insert_before(b)

        b = STORE_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        last.insert_before(b)

        # JUMP to COMPARE_OP is already part of the bytecodes
        last.setTarget(body_loc)


    def stack_eval(self, func, stack):
        #self.result = func.getOrNewRegister(self.loop_var)
        #stack.push(self.result)
        pass

    def translate(self, module, builder):
        pass

    def type_eval(self, func):
        #self.result.unify_type(int, self.debuginfo)
        pass

class STORE_SUBSCR(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(3)
    def stack_eval(self, func, stack):
        #import pdb; pdb.set_trace()
        self.result = None

    def type_eval(self, func):
        pass

    def translate(self, module, builder):
        ## for structs
        # insert_element(self, vec_val, elt_val, idx_val, name='')¶
        #builder.insert_element(self.args[1].llvm, self.args[0].llvm, self.args[2].llvm)
        #builder.insert_value(self.args[1].llvm, self.args[0].llvm, self.args[2].llvm)

        #logging.debug("Args:      {0}".format(self.args))
        #logging.debug("Arg types: {0}".format([a.type for a in self.args]))
        #logging.debug("Arg llvm:  {0}".format([str(a.llvm) for a in self.args]))
        #pdb.set_trace()
        p = builder.gep(self.args[1].llvm, [tp.Int.constant(0), self.args[2].llvm], inbounds=True)
        #logging.debug("gep:       {0}".format(str(p)))
        #builder.store(llvm.core.Type.pointer(self.args[0].llvm), p)
        builder.store(self.args[0].llvm, p)

class BINARY_SUBSCR(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        self.result = Register(func)
        stack.push(self.result)

    def type_eval(self, func):
        if not isinstance(self.args[0].type, tp.ArrayType):
            raise TypingError("Expected an array, but got {0}".format(self.args[0].type))
        self.result.unify_type(self.args[0].type.getElementType(), self.debuginfo)

    def translate(self, module, builder):
        p = builder.gep(self.args[0].llvm, [tp.Int.constant(0), self.args[1].llvm], inbounds=True)
        self.result.llvm = builder.load(p)

class POP_TOP(Bytecode):
    discard = True
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        pass

    def translate(self, module, builder):
        pass

class DUP_TOP_TWO(Bytecode):
    discard = True
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self.args[0])
        stack.push(self.args[1])
        stack.push(self.args[0])
        stack.push(self.args[1])

    def type_eval(self, func):
        pass

    def translate(self, module, builder):
        pass

class ROT_THREE(Bytecode, Poison):
    discard = True
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(3)
    def stack_eval(self, func, stack):
        stack.push(self.args[0])
        stack.push(self.args[2])
        stack.push(self.args[1])

    def type_eval(self, func):
        pass

    def translate(self, module, builder):
        pass

#---

opconst = {}
# Get all contrete subclasses of Bytecode and register them
for name in dir(sys.modules[__name__]):
    obj = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(obj, Bytecode) and len(obj.__abstractmethods__) == 0:
            opconst[dis.opmap[name]] = obj
    except TypeError:
        pass

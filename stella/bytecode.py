import dis
import sys
import types
from abc import abstractproperty

from . import tp
from . import exc
from . import utils
from . import ir
from .storage import Register, StackLoc, GlobalVariable
from .tp import Cast, Const
from .intrinsics import Intrinsic


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

            self.stack_args = args
            return f(self, func, stack)
        return extract_from_stack
    return extract_n


class Poison(object):

    """
    Require that this bytecode is rewritten by bailing out
    if it is ever evaluated.

    Note that if the child overrides all methods, this mixin will be useless
    and should be removed from the child.
    """

    def stack_eval(self, func, stack):
        raise exc.UnimplementedError(
            "{0} must be rewritten".format(
                self.__class__.__name__))

    def translate(self, cge):
        raise exc.UnimplementedError(
            "{0} must be rewritten".format(
                self.__class__.__name__))

    def type_eval(self, func):
        raise exc.UnimplementedError(
            "{0} must be rewritten".format(
                self.__class__.__name__))


class Bytecode(ir.IR):

    """
    Parent class for all Python bytecodes
    """
    pass


class ResultOnlyBytecode(Poison, ir.IR):
    """Only use this to inject values on the stack which did not originate from
    any real bytecode. This will only work at the beginning of a program
    because otherwise the bytecode may be used as the origin of a branch.
    """
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)


class LOAD_FAST(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addLocalName(self, func, name):
        # TODO: crude?
        try:
            self.args.append(func.getRegister(name))
        except exc.UndefinedError:
            self.args.append(func.getStackLoc(name))

    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        arg_type = self.args[0].type
        if self.result is None:
            type_ = type(self.args[0])
            if type_ == StackLoc:
                self.result = Register(func.impl)
            elif type_ == Register:
                self.result = self.args[0]
            else:
                raise exc.StellaException(
                    "Invalid LOAD_FAST argument type `{0}'".format(type_))
        if type(self.args[0]) == StackLoc:
            if arg_type.isReference():
                    arg_type = arg_type.dereference()
            self.result.unify_type(arg_type, self.debuginfo)

    def translate(self, cge):
        type_ = type(self.args[0])
        if type_ == StackLoc:
            self.result.llvm = cge.builder.load(self.args[0].translate(cge))
        elif type_ == Register:
            # nothing to load, it's a pseudo instruction in this case
            pass


class STORE_FAST(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.new_allocate = False

    def addLocalName(self, func, name):
        # Python does not allocate new names, it just refers to them
        (var, self.new_allocate) = func.getOrNewStackLoc(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        self.grab_stack()
        # func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        if self.result is None:
            self.result = self.popFirstArg()

        arg = self.args[0]
        if arg.type.complex_on_stack or arg.type.on_heap:
            type_ = tp.Reference(arg.type)
        else:
            type_ = arg.type
        widened, needs_cast = self.result.unify_type(type_, self.debuginfo)
        if widened:
            # TODO: can I avoid a retype in some cases?
            func.retype()
        if needs_cast:
            self.args[0] = Cast(arg, self.result.type)

    def translate(self, cge):
        self.cast(cge)
        arg = self.args[0]
        if self.new_allocate:
            type_ = self.result.type
            if type_.on_heap:
                type_ = type_.dereference()
            llvm_type = type_.llvmType(cge.module)
            self.result.llvm = cge.builder.alloca(llvm_type, name=self.result.name)
        cge.builder.store(arg.translate(cge), self.result.translate(cge))


class STORE_GLOBAL(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        # Python does not allocate new names, it just refers to them
        try:
            var = func.loadGlobal(name)
        except exc.UndefinedError:
            var = func.newGlobal(name)

        self.args.append(var)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        self.grab_stack()
        # func.retype(self.result.unify_type(self.args[1].type, self.debuginfo))
        if self.result is None:
            self.result = self.popFirstArg()
        arg = self.args[0]

        if self.result.initial_value is None:
            # This means we're defining a new variable
            self.result.setInitialValue(arg)

        widened, needs_cast = self.result.unify_type(arg.type, self.debuginfo)
        if widened:
            # TODO: can I avoid a retype in some cases?
            func.retype()
        if needs_cast:
            # TODO: side effect! Maybe that's for the best.
            self.args[0] = Cast(arg, self.result.type)

    def translate(self, cge):
        # Assume that the global has been allocated already.
        self.cast(cge)
        cge.builder.store(self.args[0].translate(cge), self.result.translate(cge))


class LOAD_CONST(Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        if self.result is None:
            self.result = self.popFirstArg()

    def translate(self, cge):
        pass


class BinaryOp(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        for i in range(len(self.args)):
            arg = self.args[i]
            widened, needs_cast = self.result.unify_type(arg.type, self.debuginfo)
            if widened:
                # TODO: can I avoid a retype in some cases?
                # It could definitely be smarter and retype the other parameter
                # directly if need be.
                func.retype()
            if needs_cast:
                self.args[i] = Cast(arg, self.result.type)

    def builderFuncName(self):
        try:
            return self.b_func[self.result.type]
        except KeyError:
            raise exc.TypeError(
                "{0} does not yet implement type {1}".format(
                    self.__class__.__name__,
                    self.result.type))

    def translate(self, cge):
        self.cast(cge)
        f = getattr(cge.builder, self.builderFuncName())
        self.result.llvm = f(
            self.args[0].translate(cge),
            self.args[1].translate(cge))

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
    b_func = {tp.Float: 'llvm.pow', tp.Int: 'llvm.powi'}

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        # TODO if args[1] is int but negative, then the result will be float, too!
        super().type_eval(func)

    def translate(self, cge):
        # llvm.pow[i]'s first argument always has to be float
        arg = self.args[0]
        if arg.type == tp.Int:
            self.args[0] = Cast(arg, tp.Float)

        self.cast(cge)

        if self.args[1].type == tp.Int:
            # powi takes a i32 argument
            power = cge.builder.trunc(
                self.args[1].translate(cge),
                tp.tp_int32,
                '(i32)' +
                self.args[1].name)
        else:
            power = self.args[1].translate(cge)

        llvm_pow = cge.module.llvm.declare_intrinsic(self.b_func[self.args[1].type],
                                                     [self.args[0].llvmType(cge.module)])
        pow_result = cge.builder.call(llvm_pow, [self.args[0].translate(cge), power])

        if isinstance(self.args[0], Cast) and \
                self.args[0].obj.type == tp.Int and self.args[1].type == tp.Int:
            # cast back to an integer
            self.result.llvm = cge.builder.fptosi(pow_result, tp.Int.llvmType(cge.module))
        else:
            self.result.llvm = pow_result


class BINARY_FLOOR_DIVIDE(BinaryOp):
    """Python compliant `//' operator.

    Slow since it has to perform type conversions and floating point division for integers"""
    b_func = {
        tp.Float: 'fdiv',
        tp.Int: 'fdiv'}  # NOT USED, but required to make it a concrete class

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def type_eval(self, func):
        self.grab_stack()
        for arg in self.args:
            # TODO: this is a HACK
            if isinstance(arg, Cast):
                type_ = arg.obj.type
            else:
                type_ = arg.type
            self.result.unify_type(type_, self.debuginfo)

        # convert all arguments to float, since fp division is required to
        # apply floor
        for i in range(len(self.args)):
            arg = self.args[i]
            do_cast = arg.type != tp.Float
            if do_cast:
                self.args[i] = Cast(arg, tp.Float)
                func.retype()

    def translate(self, cge):
        self.cast(cge)

        tmp = cge.builder.fdiv(
            self.args[0].translate(cge),
            self.args[1].translate(cge))
        llvm_floor = cge.module.llvm.declare_intrinsic('llvm.floor',
                                                       [tp.Float.llvmType(cge.module)])
        self.result.llvm = cge.builder.call(llvm_floor, [tmp])

        # TODO this is peaking too deeply into the cast
        if all([isinstance(a, Cast) and a.obj.type == tp.Int for a in self.args]):
            # TODO this may be superflous if both args got converted to float
            # in the translation stage -> move toFloat partially to the
            # analysis stage.
            self.result.llvm = cge.builder.fptosi(
                self.result.translate(cge),
                tp.Int.llvmType(cge.module),
                "(int)" +
                self.result.name)


class BINARY_TRUE_DIVIDE(BinaryOp):
    b_func = {tp.Float: 'fdiv'}

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        # The result of `/', true division, is always a float
        self.result.type = tp.Float
        super().type_eval(func)


class INPLACE_ADD(BINARY_ADD):
    pass


class INPLACE_SUBTRACT(BINARY_SUBTRACT):
    pass


class INPLACE_MULTIPLY(BINARY_MULTIPLY):
    pass


class INPLACE_TRUE_DIVIDE(BINARY_TRUE_DIVIDE):
    pass


class INPLACE_FLOOR_DIVIDE(BINARY_FLOOR_DIVIDE):
    pass


class INPLACE_MODULO(BINARY_MODULO):
    pass


class COMPARE_OP(Bytecode):
    b_func = {tp.Float: 'fcmp_ordered', tp.Int: 'icmp_signed', tp.Bool: 'icmp_signed'}
    op = None

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    def addCmp(self, op):
        self.op = op

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        self.result.type = tp.Bool
        if (self.args[0].type != self.args[1].type and
                self.args[0].type != tp.NoType and self.args[1].type != tp.NoType):
            raise exc.TypeError(
                "Comparing different types ({0} with {1})".format(
                    self.args[0].type,
                    self.args[1].type))

    def translate(self, cge):
        # assume both types are the same, see @stack_eval
        type_ = self.args[0].type
        if not self.args[0].type in self.b_func:
            raise exc.UnimplementedError(type_)

        f = getattr(cge.builder, self.b_func[type_])

        self.result.llvm = f(self.op,
                             self.args[0].translate(cge),
                             self.args[1].translate(cge))


class RETURN_VALUE(utils.BlockTerminal, Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        self.grab_stack()
        if self.result is None:
            self.result = self.popFirstArg()
        for arg in self.args:
            func.retype(self.result.unify_type(arg.type, self.debuginfo))

    def translate(self, cge):
        if self.result.type == tp.Void:
            # return None == void, do not generate a ret instruction as that is
            # invalid
           cge.builder.ret_void()
        else:
           cge.builder.ret(self.result.translate(cge))


class HasTarget(object):
    target_label = None
    target_bc = None

    def setTargetBytecode(self, bc):
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


class Jump(utils.BlockTerminal, HasTarget, ir.IR):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def processFallThrough(self):
        return False

    def stack_eval(self, func, stack):
        return [(self.target_bc, stack)]

    def type_eval(self, func):
        self.grab_stack()
        pass

    def translate(self, cge):
       cge.builder.branch(self.target_bc.block)


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
        stack.push(self.stack_args[0])
        r.append((self.target_bc, stack))
        # else continue with the next instruction (and keep the popped value)
        r.append((self.next, stack2))

        return r


class JUMP_IF_FALSE_OR_POP(Jump_if_X_or_pop, Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, cge):
       cge.builder.cbranch(
            self.args[0].translate(cge),
            self.next.block,
            self.target_bc.block)


class JUMP_IF_TRUE_OR_POP(Jump_if_X_or_pop, Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, cge):
       cge.builder.cbranch(
            self.args[0].translate(cge),
            self.target_bc.block,
            self.next.block)


class Pop_jump_if_X(Jump):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.additional_pops = 0

    def processFallThrough(self):
        self.fallthrough = self.next
        return True

    def updateTargetBytecode(self, old_bc, new_bc):
        if old_bc == self.target_bc:
            self.setTargetBytecode(new_bc)
        else:
            assert self.fallthrough == old_bc
            self.fallthrough = new_bc

    def additionalPop(self, i):
        """Deviate from Python semantics: pop i more items off the stack WHEN jumping.

        Instead of the Python semantics to pop one value of the stack, pop i more when jumping.
        """
        self.additional_pops = i

    @pop_stack(1)
    def stack_eval(self, func, stack):
        r = []
        # if X, jump
        jump_stack = stack.clone()
        for i in range(self.additional_pops):
            jump_stack.pop()
        r.append((self.target_bc, jump_stack))
        # else continue to the next instruction
        r.append((self.next, stack))
        # (pop happens in any case)

        return r


class POP_JUMP_IF_FALSE(Pop_jump_if_X, Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, cge):
       cge.builder.cbranch(
            self.args[0].translate(cge),
            self.next.block,
            self.target_bc.block)


class POP_JUMP_IF_TRUE(Pop_jump_if_X, Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def translate(self, cge):
       cge.builder.cbranch(
            self.args[0].translate(cge),
            self.target_bc.block,
            self.next.block)


class SETUP_LOOP(utils.BlockStart, HasTarget, Bytecode):
    """
    Will either be rewritten (for loop) or has no effect other than mark the
    start of a block (while loop).
    """
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, cge):
        pass

    def type_eval(self, func):
        pass


class POP_BLOCK(utils.BlockEnd, Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def stack_eval(self, func, stack):
        pass

    def translate(self, cge):
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
        stack.push(self)
    def translate(self, cge):
        if isinstance(self.var, ir.FunctionRef):
            pass
        elif isinstance(self.var, GlobalVariable):
            self.result.llvm =cge.builder.load(self.var.translate(cge))

    def type_eval(self, func):
        self.grab_stack()
        if self.result is None:
            self.var = func.impl.loadGlobal(self.args[0])
            # TODO: remove these isinstance checks and just check for
            # GlobalVariable else return directly?
            if isinstance(self.var, ir.FunctionRef):
                self.result = self.var
            elif isinstance(self.var, types.ModuleType):
                self.result = self.var
            elif isinstance(self.var, type):
                self.result = tp.PyWrapper(self.var)
            elif isinstance(self.var, Intrinsic):
                self.result = self.var
            elif isinstance(self.var, GlobalVariable):
                self.result = Register(func.impl)
            else:
                raise exc.UnimplementedError(
                    "Unknown global type {0}".format(
                        type(self.var)))

        if isinstance(self.var, GlobalVariable):
            self.result.unify_type(self.var.type, self.debuginfo)


class LOAD_ATTR(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addName(self, func, name):
        self.args.append(name)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        stack.push(self)

    def translate(self, cge):
        arg1 = self.args[1]
        if isinstance(arg1, types.ModuleType):
            return
        elif isinstance(arg1.type, tp.StructType):
            tp_attr = arg1.type.getMemberType(self.args[0])
            if isinstance(tp_attr, tp.FunctionType):
                self.result.f_self = arg1
                return
            idx = arg1.type.getMemberIdx(self.args[0])
            idx_llvm = tp.getIndex(idx)
            struct_llvm = arg1.translate(cge)
            p = cge.builder.gep(struct_llvm, [tp.Int.constant(0), idx_llvm], inbounds=True)
            self.result.llvm = cge.builder.load(p)
        else:
            raise exc.UnimplementedError(type(arg1))

    def type_eval(self, func):
        self.grab_stack()

        arg1 = self.args[1]
        if isinstance(arg1, types.ModuleType):
            self.result = func.module.loadExt(arg1, self.args[0])
            self.discard = True
        elif isinstance(arg1.type, tp.StructType):
            try:
                type_ = self.args[1].type.getMemberType(self.args[0])
                if isinstance(type_, tp.FunctionType):
                    if self.result is None:
                        self.result = func.module.getFunctionRef(type_)
                    else:
                        assert isinstance(self.result, ir.FunctionRef)
                else:
                    if self.result is None:
                        self.result = Register(func.impl)
                    self.result.unify_type(type_, self.debuginfo)
            except KeyError:
                raise exc.AttributeError("Unknown field {} of type {}".format(self.args[0],
                                                                              arg1.type),
                                         self.debuginfo)
        else:
            self.result = Register(func.impl)


class STORE_ATTR(Bytecode):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        # TODO: Does the result have to be a register? Don't I only need it for
        # the llvm propagation?
        self.result = Register(func)

    def addName(self, func, name):
        self.args.append(name)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        self.grab_stack()
        if isinstance(self.args[2].type, tp.StructType):
            member_type = self.args[2].type.getMemberType(self.args[0])
            arg_type = self.args[1].type
            if member_type != arg_type:
                if member_type == tp.Float and arg_type == tp.Int:
                    self.args[1] = tp.Cast(self.args[1], tp.Float)
                    return
                # TODO would it speed up the algorithm if arg_type is set to be
                # member_type here?
                if arg_type == tp.NoType:
                    # will be retyped anyway
                    return
                raise exc.TypeError("Argument type {} incompatible with member type {}".format(
                    arg_type, member_type))
        else:
            raise exc.UnimplementedError(
                "Cannot store attribute {0} of an object with type {1}".format(
                    self.args[0],
                    type(self.args[2])))

    def translate(self, cge):
        if (isinstance(self.args[2], tp.Typable)
              and isinstance(self.args[2].type, tp.StructType)):
            struct_llvm = self.args[2].translate(cge)
            idx = self.args[2].type.getMemberIdx(self.args[0])
            idx_llvm = tp.getIndex(idx)
            val_llvm = self.args[1].translate(cge)
            p = cge.builder.gep(struct_llvm, [tp.Int.constant(0), idx_llvm], inbounds=True)
            self.result.llvm = cge.builder.store(val_llvm, p)
        else:
            raise exc.UnimplementedError(type(self.args[2]))


class CALL_FUNCTION(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addRawArg(self, arg):
        self.num_pos_args = arg & 0xFF
        self.num_kw_args = (arg >> 8) & 0xFF
        self.num_stack_args = self.num_pos_args + self.num_kw_args*2

    def separateArgs(self):
        self.func = self.args[0]
        args = self.args[1:]

        # pdb.set_trace()
        assert len(args) == self.num_stack_args

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
        self.stack_args = []
        for i in range(self.num_pos_args + 2*self.num_kw_args + 1):
            arg = stack.pop()
            self.stack_args.append(arg)
        self.stack_args.reverse()

        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        if self.result is None:
            self.separateArgs()

            self.result = self.func.getResult(func.impl)

            if not isinstance(self.func, Intrinsic):
                func.module.functionCall(self.func, self.args, self.kw_args)

        type_ = self.func.getReturnType(self.args, self.kw_args)
        tp_change = self.result.unify_type(type_, self.debuginfo)

        if self.result.type == tp.NoType:
            # redo analysis, right now return type is not known
            func.impl.analyzeAgain()
        else:
            func.retype(tp_change)

    def translate(self, cge):
        self.result.llvm = self.func.call(
            cge,
            self.args,
            self.kw_args)


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


class ForLoop(HasTarget, ir.IR):

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def setLoopVar(self, loop_var):
        self.loop_var = loop_var

    def setLimit(self, limit):
        self.limit = limit

    def setStart(self, start):
        self.start = start

    def setEndLoc(self, end_loc):
        self.target_label = end_loc

    def setTestLoc(self, loc):
        self.test_loc = loc

    def setIterLoc(self, loc):
        """The location of FOR_ITER which may be referenced as 'restart loop'"""
        self.iter_loc = loc

    def basicSetup(self, bc):
        iter_loc = bc.loc
        start = None

        cur = bc.prev
        if not isinstance(cur, GET_ITER):
            raise exc.UnimplementedError('unsupported for loop')
        cur.remove()
        cur = bc.prev
        if not isinstance(cur, CALL_FUNCTION):
            raise exc.UnimplementedError('unsupported for loop')
        cur.remove()
        cur = bc.prev
        # TODO: this if..elif should be more general!
        if isinstance(cur, LOAD_FAST):
            limit = cur.args[0]
            cur.remove()
        elif isinstance(cur, LOAD_CONST):
            limit = cur.args[0]
            cur.remove()
        elif isinstance(cur, CALL_FUNCTION):
            cur.remove()
            limit = [cur]
            num_args = cur.num_stack_args+1  # +1 for the function name
            i = 0
            while i < num_args:
                cur = cur.prev
                # TODO: HACK. How to make this general and avoid duplicating
                # stack_eval() knowledge?
                if isinstance(cur, LOAD_ATTR):
                    # LOAD_ATTR has an argument; num_args is stack values NOT
                    # the number of bytecodes which i is counting
                    num_args +=1
                cur.remove()
                limit.append(cur)

                i += 1
        else:
            raise exc.UnimplementedError(
                'unsupported for loop: limit {0}'.format(
                    type(cur)))
        cur = bc.prev

        # this supports a start argument to range
        if isinstance(cur, LOAD_FAST) or isinstance(cur, LOAD_CONST):
            start = cur
            cur.remove()
            cur = bc.prev

        if not isinstance(cur, LOAD_GLOBAL):
            raise exc.UnimplementedError('unsupported for loop')
        cur.remove()
        cur = bc.prev
        if not isinstance(cur, SETUP_LOOP):
            raise exc.UnimplementedError('unsupported for loop')
        end_loc = cur.target_label

        self.loc = cur.loc
        # TODO set location for self and transfer jumps!
        self.setStart(start)
        self.setLimit(limit)
        self.setEndLoc(end_loc)
        self.setTestLoc(bc.loc)
        self.setIterLoc(iter_loc)

        cur.insert_after(self)
        cur.remove()

        cur = bc.next
        if not isinstance(cur, STORE_FAST):
            raise exc.UnimplementedError('unsupported for loop')
        loop_var = cur.args[0]
        self.setLoopVar(loop_var)
        cur.remove()

        bc.remove()

    def rewrite(self, func):
        last = self
        (self.limit_minus_one, _) = func.impl.getOrNewStackLoc(
            str(self.test_loc) + "__limit")

        # init
        if self.start:
            b = self.start
        else:
            b = LOAD_CONST(func.impl, self.debuginfo)
            b.addArg(Const(0))
        last.insert_after(b)
        last = b

        b = STORE_FAST(func.impl, self.debuginfo)
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
            b.addArg(self.limit)
            last.insert_after(b)
            last = b
        elif isinstance(self.limit, Const):
            b = LOAD_CONST(func.impl, self.debuginfo)
            b.addArg(self.limit)
            last.insert_after(b)
            last = b
        elif isinstance(self.limit, list):
            # limit is return value of a function call
            for b in reversed(self.limit):
                last.insert_after(b)
                last = b
            b = DUP_TOP(func.impl, self.debuginfo)
            last.insert_after(b)
            last = b

            b = ROT_THREE(func.impl, self.debuginfo)
            last.insert_after(b)
            last = b
        else:
            raise exc.UnimplementedError(
                "Unsupported limit type {0}".format(
                    type(
                        self.limit)))

        b = COMPARE_OP(func.impl, self.debuginfo)
        b.addCmp('>=')
        last.insert_after(b)
        last = b

        b = POP_JUMP_IF_TRUE(func.impl, self.debuginfo)
        b.setTarget(self.target_label)
        if isinstance(self.limit, list):
            b.additionalPop(1)
        last.insert_after(b)
        last = b

        # my_limit = limit -1
        if isinstance(self.limit, StackLoc):
            b = LOAD_FAST(func.impl, self.debuginfo)
            b.addArg(self.limit)
            last.insert_after(b)
            last = b
        elif isinstance(self.limit, Const):
            b = LOAD_CONST(func.impl, self.debuginfo)
            b.addArg(self.limit)
            last.insert_after(b)
            last = b
        elif isinstance(self.limit, list):
            # Nothing to do, the value is already on the stack
            pass
        else:
            raise exc.UnimplementedError(
                "Unsupported limit type {0}".format(
                    type(
                        self.limit)))

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
        body_loc = b.linearNext().loc
        func.addLabel(b.linearNext())

        jump_updates = []
        while b.next is not None:
            if isinstance(b, Jump) and b.target_label == self.iter_loc:
                jump_updates.append(b)
            b = b.next
        assert isinstance(b, utils.BlockEnd)
        jump_loc = b.loc
        last = b.prev
        b.remove()

        # go back to the JUMP and switch locations
        loop_test_loc = last.loc
        last.loc = jump_loc
        func.replaceLocation(last)

        for b in jump_updates:
            b.setTarget(loop_test_loc)

        if last.linearPrev().equivalent(last) and isinstance(last, JUMP_ABSOLUTE):
            # Python seems to sometimes add a duplicate JUMP_ABSOLUTE at the
            # end of the loop. Remove it.
            last.linearPrev().remove()

        # loop test
        # pdb.set_trace()
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
        b.loc = loop_test_loc
        func.replaceLocation(b)
        last.insert_before(b)

        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.limit_minus_one)
        last.insert_before(b)

        b = COMPARE_OP(func.impl, self.debuginfo)
        b.addCmp('>=')
        last.insert_before(b)

        b = POP_JUMP_IF_TRUE(func.impl, self.debuginfo)
        b.setTarget(self.target_label)
        last.insert_before(b)

        # increment
        b = LOAD_FAST(func.impl, self.debuginfo)
        b.addArg(self.loop_var)
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
        # self.result = func.getOrNewRegister(self.loop_var)
        # stack.push(self.result)
        pass

    def translate(self, cge):
        pass

    def type_eval(self, func):
        self.grab_stack()
        # self.result.unify_type(int, self.debuginfo)
        pass


class STORE_SUBSCR(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(3)
    def stack_eval(self, func, stack):
        self.result = None

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        if self.args[1].type.isReference():
            type_ = self.args[1].type.dereference()
        else:
            type_ = self.args[1].type
        type_.storeSubscript(cge, self.args[1], self.args[2], self.args[0])


class BINARY_SUBSCR(Bytecode):
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        if self.args[0].type.isReference():
            arg_type = self.args[0].type.dereference()
        else:
            arg_type = self.args[0].type
        if not isinstance(arg_type, tp.Subscriptable):
            raise exc.TypeError(
                "Type must be subscriptable, but got {0}".format(
                    self.args[0].type))
        self.result.unify_type(
            arg_type.getElementType(self.args[1]),
            self.debuginfo)

    def translate(self, cge):
        if self.args[0].type.isReference():
            type_ = self.args[0].type.dereference()
        else:
            type_ = self.args[0].type
        self.result.llvm = type_.loadSubscript(cge, self.args[0], self.args[1])


class POP_TOP(Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        pass

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        pass


class DUP_TOP(Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        stack.push(self.stack_args[0])
        stack.push(self.stack_args[0])

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        pass


class DUP_TOP_TWO(Bytecode):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self.stack_args[0])
        stack.push(self.stack_args[1])
        stack.push(self.stack_args[0])
        stack.push(self.stack_args[1])

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        pass


class ROT_TWO(Bytecode, Poison):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(2)
    def stack_eval(self, func, stack):
        stack.push(self.stack_args[1])
        stack.push(self.stack_args[0])

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        pass


class ROT_THREE(Bytecode, Poison):
    discard = True

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    @pop_stack(3)
    def stack_eval(self, func, stack):
        stack.push(self.stack_args[2])
        stack.push(self.stack_args[0])
        stack.push(self.stack_args[1])

    def type_eval(self, func):
        self.grab_stack()

    def translate(self, cge):
        pass


class UNARY_NEGATIVE(Bytecode):
    b_func = {tp.Float: 'fsub', tp.Int: 'sub'}

    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)
        self.result = Register(func)

    @pop_stack(1)
    def stack_eval(self, func, stack):
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        arg = self.args[0]
        self.result.unify_type(arg.type, self.debuginfo)

    def builderFuncName(self):
        try:
            return self.b_func[self.result.type]
        except KeyError:
            raise exc.TypeError(
                "{0} does not yet implement type {1}".format(
                    self.__class__.__name__,
                    self.result.type))

    def translate(self, cge):
        self.cast(cge)
        f = getattr(cge.builder, self.builderFuncName())
        self.result.llvm = f(
            self.result.type.constant(0),
            self.args[0].translate(cge))


class UNPACK_SEQUENCE(Bytecode):
    n = 0
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addRawArg(self, arg):
        self.n = arg

    @pop_stack(1)
    def stack_eval(self, func, stack):
        self.result = []
        for i in range(self.n):
            reg = Register(func)
            stack.push(self)
            self.result.append(reg)

    def type_eval(self, func):
        self.grab_stack()
        i = 0
        for reg in reversed(self.result):
            reg.unify_type(self.args[0].type.getElementType(i), self.debuginfo)
            i += 1

    def translate(self, cge):
        if self.args[0].type.isReference():
            type_ = self.args[0].type.dereference()
        else:
            type_ = self.args[0].type
        i = 0
        for reg in reversed(self.result):
            reg.llvm = type_.loadSubscript(cge, self.args[0], i)
            i += 1


class BUILD_TUPLE(Bytecode):
    n = 0
    def __init__(self, func, debuginfo):
        super().__init__(func, debuginfo)

    def addRawArg(self, arg):
        self.n = arg

    def stack_eval(self, func, stack):
        self.stack_args = []
        for i in range(self.n):
            self.stack_args.append(stack.pop())
        stack.push(self)

    def type_eval(self, func):
        self.grab_stack()
        if not self.result:
            self.result = tp.Tuple(self.args)
        else:
            self.result.unify_type(self.args)

    def translate(self, cge):
        self.result.translate(cge)


opconst = {}
# Get all contrete subclasses of Bytecode and register them
for name in dir(sys.modules[__name__]):
    obj = sys.modules[__name__].__dict__[name]
    try:
        if issubclass(obj, Bytecode) and len(obj.__abstractmethods__) == 0:
            opconst[dis.opmap[name]] = obj
    except TypeError:
        pass

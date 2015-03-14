from . import tp
import llvmlite.ir as ll


class Register(tp.Typable):
    name = None

    def __init__(self, func, name=None):
        super().__init__()
        if name:
            assert type(name) == str
            self.name = name
        else:
            self.name = func.newRegisterName()

    def __str__(self):
        return "{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name


class StackLoc(tp.Typable):
    name = None

    def __init__(self, func, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return "%{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name


class GlobalVariable(tp.Typable):
    name = None
    initial_value = None

    def __init__(self, name, initial_value=None):
        super().__init__()
        self.name = name
        if initial_value is not None:
            self.setInitialValue(initial_value)

    def setInitialValue(self, initial_value):
        if isinstance(initial_value, tp.Typable):
            self.initial_value = initial_value
        else:
            self.initial_value = tp.wrapValue(initial_value)
        self.type = self.initial_value.type
        self.type.makePointer(True)

    def __str__(self):
        return "+{0}<{1}>".format(self.name, self.type)

    def __repr__(self):
        return self.name

    def translate(self, cge):
        if self.llvm:
            return self.llvm

        self.llvm = ll.GlobalVariable(cge.module.llvm, self.llvmType(cge.module), self.name)
        # TODO: this condition is too complicated and likely means that my
        # code is not working consistently with the attribute
        llvm_init = None
        if (hasattr(self.initial_value, 'llvm')
                and self.initial_value is not None):
            llvm_init = self.initial_value.translate(cge)

        if llvm_init is None:
            self.llvm.initializer = ll.Constant(self.initial_value.type.llvmType(cge.module),
                                                ll.Undefined)
        else:
            self.llvm.initializer = llvm_init

        return self.llvm

import dis


class StellaException(Exception):
    def __init__(self, msg, debuginfo=None):
        super().__init__(msg)

        self.addDebug(debuginfo)

    def addDebug(self, debuginfo):
        if debuginfo:
            self.debuginfo = debuginfo

    def __str__(self):
        if hasattr(self, 'debuginfo'):
            return '{0} at {1}'.format(super().__str__(), self.debuginfo)
        else:
            return super().__str__()


class UnsupportedOpcode(StellaException):
    def __init__(self, op, debuginfo):
        super().__init__(dis.opname[op])
        self.addDebug(debuginfo)


class UnsupportedTypeError(StellaException, TypeError):
    def __init__(self, msg, debuginfo=None):
        self.name_stack = []
        self.type_stack = []
        super().__init__(msg)

        self.addDebug(debuginfo)

    def prepend(self, name, type):
        self.name_stack.append(name)
        self.type_stack.append(type)


    def __str__(self):
        fields = ".".join(reversed(self.name_stack))
        if fields:
            fields += ': '
        return fields + super().__str__()


class TypeError(StellaException, TypeError):
    def __init__(self, msg, debuginfo=None):
        super().__init__(msg)

        self.addDebug(debuginfo)


class UnimplementedError(StellaException):
    pass


class UndefinedError(StellaException):
    pass


class UndefinedGlobalError(StellaException):
    pass


class InternalError(StellaException):
    pass


class WrongNumberOfArgsError(StellaException):
    pass


class AttributeError(StellaException, AttributeError):
    def __init__(self, msg, debuginfo=None):
        super().__init__(msg)

        self.addDebug(debuginfo)


class IndexError(StellaException, IndexError):
    def __init__(self, msg, debuginfo=None):
        super().__init__(msg)

        self.addDebug(debuginfo)

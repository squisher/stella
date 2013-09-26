import dis
import inspect

class Debuginfo(object):
    line = None
    filename = None

class Bytecode(object):
    pass

def disassemble(co, lasti=-1):
    """Disassemble a code object."""
    code = co.co_code
    labels = dis.findlabels(code)
    linestarts = dict(dis.findlinestarts(co))
    n = len(code)
    i = 0
    extended_arg = 0
    free = None
    while i < n:
        op = code[i]
        if i in linestarts:
            if i > 0:
                print()
            print("%3d" % linestarts[i], end=' ')
        else:
            print('   ', end=' ')

        if i == lasti: print('-->', end=' ')
        else: print('   ', end=' ')
        if i in labels: print('>>', end=' ')
        else: print('  ', end=' ')
        print(repr(i).rjust(4), end=' ')
        print(dis.opname[op].ljust(20), end=' ')
        i = i+1
        if op >= dis.HAVE_ARGUMENT:
            oparg = code[i] + code[i+1]*256 + extended_arg
            extended_arg = 0
            i = i+2
            if op == dis.EXTENDED_ARG:
                extended_arg = oparg*65536
            print(repr(oparg).rjust(5), end=' ')
            if op in dis.hasconst:
                print('(' + repr(co.co_consts[oparg]) + ')', end=' ')
            elif op in dis.hasname:
                print('(' + co.co_names[oparg] + ')', end=' ')
            elif op in dis.hasjrel:
                print('(to ' + repr(i + oparg) + ')', end=' ')
            elif op in dis.haslocal:
                print('(' + co.co_varnames[oparg] + ')', end=' ')
            elif op in dis.hascompare:
                print('(' + cmp_op[oparg] + ')', end=' ')
            elif op in dis.hasfree:
                if free is None:
                    free = co.co_cellvars + co.co_freevars
                print('(' + free[oparg] + ')', end=' ')
        print()

class Function(object):
    f = None
    symboltable = dict()
    args = []
    return_tp = None

    def __init__(self, f):
        self.f = f
        argspec = inspect.getfullargspec(f)
        self.args = [Variable(n) for n in argspec.args]

    def analyze(self):
        disassemble(self.f.__code__)

class Variable(object):
    name = None
    tp = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name + ':' + self.tp

def main(f):
    f = Function(f)
    f.analyze()

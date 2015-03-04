from dis import dis
import stella
import llvmlite.ir as ll
import llvmlite.binding as llvm
import pdb
from pdb import pm

# try to load everything from wip.py
try:
    from wip import *
except ImportError:
    pass

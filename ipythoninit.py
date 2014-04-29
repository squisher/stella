from dis import dis
import stella
import llvm, llvm.core, llvm.ee, llvm.passes
import pdb
from pdb import pm

# try to load everything from wip.py
try:
    from wip import *
except ImportError:
    pass

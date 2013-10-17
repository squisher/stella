from dis import dis
from stella import stella
import llvm, llvm.core, llvm.ee

# try to load everything from wip.py
try:
    from wip import *
except ImportError:
    pass

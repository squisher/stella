from test.langconstr import *
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))

def current_work():
    print(stella.wrap(array_allocation, ir=True)())

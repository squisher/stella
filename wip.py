from test.langconstr import *
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))

def current_work():
    print(stella.wrap(hof_f, ir=True)(2))

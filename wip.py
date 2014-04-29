from test.langconstr import *
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))

def current_work():
    stella.wrap(hof_m, ir=True)(12.0)

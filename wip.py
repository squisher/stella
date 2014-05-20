from test.langconstr import *
import stella
#args_mod = list(filter(lambda e: e[0] >= 0, arglist2))

def current_work(run=False):
    print(stella.wrap(kwargs_call, ir=not run)(42))

from math import log

# cdef extern void c_eject_tomato "eject_tomato" (float speed)
cdef extern from "mtwist-1.1/mtwist.c":
    double c_mt_drand "mt_drand" ()
    void c_mt_seed32new "mt_seed32new" (unsigned int)

#def uniform():
#    return mt_drand()
#
#def exp(p):
#    u = 1.0 - uniform()
#    return -log(u)/p
#
#def seed(s):
#    mt_seed32new (s)

def mt_drand():
    return c_mt_drand()

def mt_seed32new(s):
    c_mt_seed32new (s)

#
# Copyright 2013-2015 David Mohr
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#from math import log
import ctypes

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

def getCSignatures():
    """There should be a way to retrieve this info from cython, but I couldn't find it"""
    return {
        'mt_drand': (ctypes.c_double, []),
        'mt_seed32new': (None, [ctypes.c_uint32])
    }

def mt_drand():
    return c_mt_drand()

def mt_seed32new(s):
    c_mt_seed32new (s)

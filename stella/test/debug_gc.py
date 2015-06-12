# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import stella
import stella.ir
from . import langconstr
# import types


def check():
    print("Collecting {}".format(gc.collect()))

    for obj in filter(lambda x: isinstance(x, stella.ir.Module), gc.get_objects()):
        print ('-'*48)
        print("{} | {} : in={}, out={}".format(
            str(obj), repr(obj), len(gc.get_referrers(obj)), len(gc.get_referrers(obj))))
        for r in gc.get_referrers(obj):
            print(" < {}".format(type(r)))
            if isinstance(r, list):
                if len(r) > 20:
                    print("  ", len(r))
                    continue
            print("  ", r)
#            for rr in gc.get_referrers(r):
#                print("   < {}".format(type(r)))
#        for r in gc.get_referents(obj):
#            print(" > {}".format(type(r)))

# with bug
r = stella.wrap(langconstr.kwargs_call1)(1)
print(r)
# with bug
#r = stella.wrap(langconstr.call_void)()
#print(r)
# no bug
#r = stella.wrap(langconstr.array_alloc_use)()
#print(r)

print ('='*78)
check()

print ('='*78)
print("Garbarge: ", len(gc.garbage), any([isinstance(x, stella.ir.Module) for x in gc.garbage]))
for m in filter(lambda x: isinstance(x, stella.ir.Module), gc.garbage):
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

print('-'*78)
check()

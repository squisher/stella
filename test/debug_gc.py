import gc
import stella
import stella.ir
import test.langconstr
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
r = stella.wrap(test.langconstr.kwargs_call1)(1)
print(r)
# with bug
#r = stella.wrap(test.langconstr.call_void)()
#print(r)
# no bug
#r = stella.wrap(test.langconstr.array_alloc_use)()
#print(r)

print ('='*78)
check()

print ('='*78)
print("Garbarge: ", len(gc.garbage), any([isinstance(x, stella.ir.Module) for x in gc.garbage]))
for m in filter(lambda x: isinstance(x, stella.ir.Module), gc.garbage):
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

print('-'*78)
check()

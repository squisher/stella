CFLAGS=-Wall -std=gnu99
CFLAGS+=-ggdb3
#CFLAGS+=-O2
LDFLAGS=-lm

CPPFLAGS=-Dlint

CXXFLAGS=-Wall -std=gnu++0x
CXXFLAGS+=-ggdb3

lib%.o : %.c
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

EXECUTABLES=mtpy.so

all: mtpy

mtpy:
	./mtpy_setup.py build_ext --inplace

virtnet_c: virtnet_c.o
virtnet_c.o: virtnet_c.c
mtwist-1.1/mtwist.o: mtwist-1.1/mtwist.c
mtwist-1.1/libmtwist.o: mtwist-1.1/mtwist.c

libmtwist.so: mtwist-1.1/libmtwist.o
	${CC} ${LDFLAGS} ${LDLIBS} -shared -Wl,-soname,libmtwist.so.1.1 -o $@ $^
clean:
	rm *.o *.so ${EXECUTABLES}

CFLAGS:=-Wall -O1
#CC=gcc
CC=clang

%.s: %.c
	${CC} ${CFLAGS} -emit-llvm -S $<

ctest: ctest.o
ctest.o: ctest.c
ctest.s: ctest.c

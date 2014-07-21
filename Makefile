CFLAGS:=-Wall -O0
#CC=gcc
CC=clang

%.s: %.c
	rm -f $@.tmp
	${CC} ${CFLAGS} -O0 -emit-llvm -S $<
	echo "\n\n;;; O0" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O1 -emit-llvm -S $<
	echo "\n\n;;; O1" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O2 -emit-llvm -S $<
	echo "\n\n;;; O2" >> $@.tmp
	cat $@ >> $@.tmp
	mv $@.tmp $@

%.llc.cpp: %.s
	llc -march=cpp -o $@ $<

ctest: ctest.o
ctest.o: ctest.c
ctest.s: ctest.c

fib: fib.o
fib.o: fib.c
